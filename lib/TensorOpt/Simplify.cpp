#include "TensorOpt/Passes.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::topt {
#define GEN_PASS_DEF_TOPTSIMPLIFY
#include "TensorOpt/Passes.h.inc"

namespace {

bool isIdentity(ArrayRef<int64_t> perm) {
  for (auto [i, p] : llvm::enumerate(perm))
    if (static_cast<int64_t>(i) != p)
      return false;
  return true;
}

/// transpose(transpose(x, p), q) -> transpose(x, r), r[k] = p[q[k]].
///
/// Why: with dim(y, i) = dim(x, p[i]) and dim(z, k) = dim(y, q[k]), output
/// dim k of z walks input dim p[q[k]] of x. Both ops only move elements, so
/// the pair is one move, and when r is the identity it moves nothing.
/// The inner transpose is left in place; if it has no other users the
/// greedy driver erases it as dead, and if it does, it was needed anyway.
struct ComposeTransposes : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp outer,
                                PatternRewriter &rewriter) const override {
    auto inner = outer.getInput().getDefiningOp<TransposeOp>();
    if (!inner)
      return failure();
    ArrayRef<int64_t> p = inner.getPermutation(), q = outer.getPermutation();
    SmallVector<int64_t> r(q.size());
    for (auto [k, qk] : llvm::enumerate(q))
      r[k] = p[qk];
    if (isIdentity(r)) {
      rewriter.replaceOp(outer, inner.getInput());
      return success();
    }
    rewriter.replaceOpWithNewOp<TransposeOp>(outer, outer.getType(),
                                             inner.getInput(), r);
    return success();
  }
};

/// transpose(x, identity) -> x.
struct DropIdentityTranspose : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (!isIdentity(op.getPermutation()))
      return failure();
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
};

/// If `v` is a splat f32 constant, return its value.
std::optional<APFloat> getSplat(Value v) {
  DenseFPElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)) || !attr.isSplat())
    return std::nullopt;
  return attr.getSplatValue<APFloat>();
}

/// x * 1.0 -> x (either operand order).
///
/// Exact for every x: IEEE multiplication by one returns x, including -0.0,
/// infinities and quiet NaNs. (A signalling NaN would be quietened by the
/// multiply and not by the rewrite; nothing in this dialect can produce one.)
struct MulByOne : OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    for (auto [self, other] : {std::pair(op.getRhs(), op.getLhs()),
                               std::pair(op.getLhs(), op.getRhs())}) {
      auto c = getSplat(self);
      if (c && c->isExactlyValue(1.0)) {
        rewriter.replaceOp(op, other);
        return success();
      }
    }
    return failure();
  }
};

/// x + (-0.0) -> x (either operand order).
///
/// -0.0 is the additive identity of IEEE addition under round-to-nearest:
/// (+0) + (-0) = +0, (-0) + (-0) = -0, and every other x is unchanged.
/// +0.0 is NOT: (-0) + (+0) = +0, so x + 0.0 -> x would flip the sign of
/// a negative zero. That rewrite is intentionally not implemented.
struct AddNegativeZero : OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    for (auto [self, other] : {std::pair(op.getRhs(), op.getLhs()),
                               std::pair(op.getLhs(), op.getRhs())}) {
      auto c = getSplat(self);
      if (c && c->isZero() && c->isNegative()) {
        rewriter.replaceOp(op, other);
        return success();
      }
    }
    return failure();
  }
};

struct SimplifyPass : impl::ToptSimplifyBase<SimplifyPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ComposeTransposes, DropIdentityTranspose, MulByOne,
                 AddNegativeZero>(&getContext());
    // The driver folds by default. Turn that off, or this pass would also be
    // a constant folder and the benchmark could not separate the two.
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
