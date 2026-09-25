#include "TensorOpt/Passes.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::topt {
#define GEN_PASS_DEF_TOPTFUSEELEMENTWISE
#include "TensorOpt/Passes.h.inc"

namespace {

bool isElementwise(Operation *op) {
  return isa_and_nonnull<AddOp, MulOp, FusedElementwiseOp>(op);
}

/// Emit the scalar computation of elementwise `op` applied to `args`, one
/// f32 per tensor operand of `op`, and return the scalar result.
Value emitScalarBody(Operation *op, OpBuilder &b, Location loc,
                     ArrayRef<Value> args) {
  if (isa<AddOp>(op))
    return arith::AddFOp::create(b, loc, args[0], args[1]);
  if (isa<MulOp>(op))
    return arith::MulFOp::create(b, loc, args[0], args[1]);
  auto fused = cast<FusedElementwiseOp>(op);
  Block &body = fused.getBody().front();
  IRMapping map;
  map.map(body.getArguments(), args);
  for (Operation &inner : body.without_terminator())
    b.clone(inner, map);
  return map.lookup(cast<YieldOp>(body.getTerminator()).getValue());
}

/// consumer(..., producer(xs), ...) -> fused_elementwise(xs ∪ others)
///
/// Why this is correct. Every elementwise op here computes
/// out[i] = f(in_0[i], ..., in_n[i]) for all i in one shared index space
/// (SameOperandsAndResultType: no broadcasting, no indexing maps). So
/// consumer(producer(x))[i] = g(f(x[i])): evaluating f and then g on the
/// same scalar, per index, is the same function. The ops are Pure, so
/// nothing can observe that the intermediate tensor never exists.
/// Requiring the consumer to be the producer's only user is what makes the
/// producer safe to erase without recomputing it anywhere.
///
/// Floating point is untouched: the scalar ops are emitted in the original
/// order, with no reassociation, so fused and unfused results are
/// bit-identical (the benchmark checks this).
struct FuseIntoConsumer : RewritePattern {
  FuseIntoConsumer(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *consumer,
                                PatternRewriter &rewriter) const override {
    if (!isElementwise(consumer))
      return failure();

    Operation *producer = nullptr;
    for (Value v : consumer->getOperands()) {
      Operation *def = v.getDefiningOp();
      if (isElementwise(def) &&
          llvm::all_of(def->getUsers(),
                       [&](Operation *u) { return u == consumer; })) {
        producer = def;
        break;
      }
    }
    if (!producer)
      return rewriter.notifyMatchFailure(consumer, "no single-use producer");
    Value produced = producer->getResult(0);

    // Inputs of the fused op: the producer's operands and the consumer's
    // other operands, each tensor once even if it is read several times.
    llvm::SetVector<Value> inputs;
    inputs.insert(producer->operand_begin(), producer->operand_end());
    for (Value v : consumer->getOperands())
      if (v != produced)
        inputs.insert(v);

    Location loc = rewriter.getFusedLoc({producer->getLoc(),
                                         consumer->getLoc()});
    Type type = consumer->getResult(0).getType();
    auto fused =
        FusedElementwiseOp::create(rewriter, loc, type, inputs.getArrayRef());

    Type f32 = rewriter.getF32Type();
    SmallVector<Type> argTypes(inputs.size(), f32);
    SmallVector<Location> argLocs(inputs.size(), loc);
    Block *body = rewriter.createBlock(&fused.getBody(), {}, argTypes,
                                       argLocs);
    auto scalarOf = [&](Value tensor) {
      return body->getArgument(
          std::distance(inputs.begin(), llvm::find(inputs, tensor)));
    };

    SmallVector<Value> producerArgs;
    for (Value v : producer->getOperands())
      producerArgs.push_back(scalarOf(v));
    Value p = emitScalarBody(producer, rewriter, loc, producerArgs);

    SmallVector<Value> consumerArgs;
    for (Value v : consumer->getOperands())
      consumerArgs.push_back(v == produced ? p : scalarOf(v));
    Value c = emitScalarBody(consumer, rewriter, loc, consumerArgs);
    YieldOp::create(rewriter, loc, c);

    rewriter.replaceOp(consumer, fused.getResult());
    rewriter.eraseOp(producer);
    return success();
  }
};

struct FuseElementwisePass
    : impl::ToptFuseElementwiseBase<FuseElementwisePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseIntoConsumer>(&getContext());
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::topt
