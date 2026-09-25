#include "TensorOpt/Passes.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::topt {
#define GEN_PASS_DEF_CONVERTTOPTTOLINALG
#include "TensorOpt/Passes.h.inc"

namespace {

using ScalarFn =
    function_ref<Value(OpBuilder &, Location, ArrayRef<Value> scalars)>;

/// Build a linalg.generic that evaluates `scalar` at every index of the
/// result, reading one f32 per operand. All maps are identities: in topt,
/// out[i] depends on in_k[i] only.
Value buildElementwiseGeneric(ConversionPatternRewriter &rewriter,
                              Location loc, RankedTensorType type,
                              ValueRange operands, ScalarFn scalar) {
  // A splat operand is passed as a scalar constant, not as a tensor input.
  SmallVector<std::optional<FloatAttr>> splats;
  SmallVector<Value> tensorInputs;
  for (Value v : operands) {
    DenseFPElementsAttr attr;
    if (matchPattern(v, m_Constant(&attr)) && attr.isSplat()) {
      splats.push_back(
          FloatAttr::get(type.getElementType(), attr.getSplatValue<APFloat>()));
    } else {
      splats.push_back(std::nullopt);
      tensorInputs.push_back(v);
    }
  }

  Value init = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                       type.getElementType());
  SmallVector<AffineMap> maps(
      tensorInputs.size() + 1,
      rewriter.getMultiDimIdentityMap(type.getRank()));
  SmallVector<utils::IteratorType> iterators(type.getRank(),
                                             utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{type}, tensorInputs, ValueRange{init}, maps,
      iterators, [&](OpBuilder &b, Location l, ValueRange args) {
        SmallVector<Value> scalars;
        unsigned next = 0;
        for (auto &splat : splats)
          scalars.push_back(splat ? arith::ConstantOp::create(b, l, *splat)
                                        .getResult()
                                  : args[next++]);
        linalg::YieldOp::create(b, l, scalar(b, l, scalars));
      });
  return generic.getResult(0);
}

struct LowerConstant : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

template <typename SourceOp, typename ArithOp>
struct LowerBinary : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value r = buildElementwiseGeneric(
        rewriter, op.getLoc(), op.getType(), adaptor.getOperands(),
        [](OpBuilder &b, Location l, ArrayRef<Value> s) -> Value {
          return ArithOp::create(b, l, s[0], s[1]);
        });
    rewriter.replaceOp(op, r);
    return success();
  }
};

struct LowerFused : OpConversionPattern<FusedElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FusedElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();
    Value r = buildElementwiseGeneric(
        rewriter, op.getLoc(), op.getOutput().getType(), adaptor.getInputs(),
        [&](OpBuilder &b, Location, ArrayRef<Value> s) -> Value {
          IRMapping map;
          map.map(body.getArguments(), s);
          for (Operation &inner : body.without_terminator())
            b.clone(inner, map);
          return map.lookup(cast<YieldOp>(body.getTerminator()).getValue());
        });
    rewriter.replaceOp(op, r);
    return success();
  }
};

struct LowerTranspose : OpConversionPattern<TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Same permutation convention as linalg.transpose, so it maps 1:1.
    auto type = op.getOutput().getType();
    Value init = tensor::EmptyOp::create(rewriter, op.getLoc(),
                                         type.getShape(),
                                         type.getElementType());
    auto t = linalg::TransposeOp::create(rewriter, op.getLoc(),
                                         adaptor.getInput(), init,
                                         op.getPermutation());
    rewriter.replaceOp(op, t.getResult());
    return success();
  }
};

struct LowerMatmul : OpConversionPattern<MatmulOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // linalg.matmul accumulates into its output: C += A * B. Start at +0.0.
    // (The constant folder starts its accumulator at +0.0 too, and must.)
    Location loc = op.getLoc();
    auto type = op.getOutput().getType();
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(type.getElementType(), 0.0));
    Value init = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                         type.getElementType());
    Value filled = linalg::FillOp::create(rewriter, loc, TypeRange{type},
                                          ValueRange{zero}, ValueRange{init})
                       .getResult(0);
    auto mm = linalg::MatmulOp::create(
        rewriter, loc, TypeRange{type},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{filled});
    rewriter.replaceOp(op, mm.getResult(0));
    return success();
  }
};

struct ConvertToptToLinalgPass
    : impl::ConvertToptToLinalgBase<ConvertToptToLinalgPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect>();
    target.addIllegalDialect<TensorOptDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<LowerConstant, LowerBinary<AddOp, arith::AddFOp>,
                 LowerBinary<MulOp, arith::MulFOp>, LowerFused,
                 LowerTranspose, LowerMatmul>(ctx);
    // Rejected: applyFullConversion. It would also demand that func.func and
    // func.return be legal-by-declaration; partial conversion with topt
    // marked illegal gives the same guarantee for the ops this pass owns.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    // Splat constants now consumed as scalars leave dead tensor constants
    // behind. Erase them here rather than run -canonicalize, which would
    // also apply arith's own x*1 and x+(-0) folds and blur which topt pass
    // saved what.
    SmallVector<Operation *> dead;
    getOperation().walk([&](arith::ConstantOp c) {
      if (c->use_empty())
        dead.push_back(c);
    });
    for (Operation *op : dead)
      op->erase();
  }
};

} // namespace
} // namespace mlir::topt
