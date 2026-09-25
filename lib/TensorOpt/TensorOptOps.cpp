#include "TensorOpt/TensorOptDialect.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::topt;

#define GET_OP_CLASSES
#include "TensorOpt/TensorOptOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Constant folding
//
// Every fold() here evaluates an op whose operands are all compile-time
// constants, and does nothing else. Algebraic identities such as
// transpose(transpose(x)) -> x live in the simplification pass instead, so
// the two can be switched on independently and measured separately.
// Rejected: the upstream idiom of putting identities in fold() too. Then
// -topt-constant-fold would also simplify, and the benchmark could not
// attribute a saving to either one.
//
// Two rules every folder obeys:
//
//  1. Round exactly like the generated code. Arithmetic goes through APFloat
//     in IEEE single precision, round-to-nearest-even, one rounding per
//     operation, in the same order as the lowered loops. Host `float` math
//     would do in C++ what the compiler is free to contract into an fma, and
//     the folded constant would differ in the last bit from the value the
//     unfolded program computes.
//  2. Do not trade a cheap runtime op for a huge binary. A folded non-splat
//     result is stored in .rodata, so folding is capped by element count, and
//     matmul also by multiply-accumulates (APFloat is slow).
//===----------------------------------------------------------------------===//

namespace {
constexpr int64_t kMaxFoldedElements = int64_t(1) << 18; // 1 MiB of f32
constexpr int64_t kMaxFoldedMatmulMacs = int64_t(1) << 20;

bool withinFoldBudget(ShapedType type, ArrayRef<Attribute> operands) {
  bool allSplat = llvm::all_of(operands, [](Attribute a) {
    auto e = dyn_cast_or_null<DenseElementsAttr>(a);
    return e && e.isSplat();
  });
  // A splat result costs one scalar however large the tensor is.
  return allSplat || type.getNumElements() <= kMaxFoldedElements;
}
} // namespace

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  if (!withinFoldBudget(getType(), adaptor.getOperands()))
    return {};
  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(), getType(),
      [](const APFloat &a, const APFloat &b) { return a + b; });
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  if (!withinFoldBudget(getType(), adaptor.getOperands()))
    return {};
  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(), getType(),
      [](const APFloat &a, const APFloat &b) { return a * b; });
}

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto input = dyn_cast_or_null<DenseFPElementsAttr>(adaptor.getInput());
  if (!input)
    return {};
  auto outType = getOutput().getType();
  // Every element is the same, so moving them around changes nothing.
  if (input.isSplat())
    return input.resizeSplat(outType);
  if (outType.getNumElements() > kMaxFoldedElements)
    return {};

  ArrayRef<int64_t> perm = getPermutation();
  ArrayRef<int64_t> inShape = input.getType().getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  int64_t rank = outType.getRank();

  // Row-major strides of the input, then out[idx] = in[idx'] where
  // idx'[perm[i]] = idx[i]: output dim i walks input dim perm[i].
  SmallVector<int64_t> inStride(rank, 1);
  for (int64_t d = rank - 2; d >= 0; --d)
    inStride[d] = inStride[d + 1] * inShape[d + 1];

  auto values = llvm::to_vector(input.getValues<APFloat>());
  SmallVector<APFloat> out;
  out.reserve(outType.getNumElements());
  SmallVector<int64_t> idx(rank, 0);
  for (int64_t n = 0, e = outType.getNumElements(); n < e; ++n) {
    int64_t src = 0;
    for (int64_t i = 0; i < rank; ++i)
      src += idx[i] * inStride[perm[i]];
    out.push_back(values[src]);
    for (int64_t i = rank - 1; i >= 0; --i) { // odometer increment
      if (++idx[i] < outShape[i])
        break;
      idx[i] = 0;
    }
  }
  return DenseFPElementsAttr::get(outType, out);
}

OpFoldResult MatmulOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<DenseFPElementsAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<DenseFPElementsAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};
  int64_t m = lhs.getType().getDimSize(0), k = lhs.getType().getDimSize(1),
          n = rhs.getType().getDimSize(1);
  if (m * n * k > kMaxFoldedMatmulMacs ||
      m * n > kMaxFoldedElements)
    return {};

  auto a = llvm::to_vector(lhs.getValues<APFloat>());
  auto b = llvm::to_vector(rhs.getValues<APFloat>());
  const llvm::fltSemantics &sem = APFloat::IEEEsingle();
  SmallVector<APFloat> c;
  c.reserve(m * n);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      // Same order as the lowered code: the accumulator starts at +0.0
      // (linalg.fill) and k is the innermost loop, one rounding per step.
      APFloat acc = APFloat::getZero(sem);
      for (int64_t p = 0; p < k; ++p) {
        APFloat prod = a[i * k + p];
        prod.multiply(b[p * n + j], APFloat::rmNearestTiesToEven);
        acc.add(prod, APFloat::rmNearestTiesToEven);
      }
      c.push_back(acc);
    }
  return DenseFPElementsAttr::get(getOutput().getType(), c);
}
