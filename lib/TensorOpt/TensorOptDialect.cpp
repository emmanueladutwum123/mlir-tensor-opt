#include "TensorOpt/TensorOptDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::topt;

#include "TensorOpt/TensorOptOpsDialect.cpp.inc"

void TensorOptDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TensorOpt/TensorOptOps.cpp.inc"
      >();
}

Operation *TensorOptDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  auto elements = dyn_cast<DenseFPElementsAttr>(value);
  if (!elements || elements.getType() != type)
    return nullptr;
  return ConstantOp::create(builder, loc, elements);
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto out = getOutput().getType();
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || out.getRank() != 2)
    return emitOpError("expects rank-2 operands and result");
  if (lhs.getDimSize(1) != rhs.getDimSize(0))
    return emitOpError("contraction dimensions differ: lhs has ")
           << lhs.getDimSize(1) << " columns, rhs has " << rhs.getDimSize(0)
           << " rows";
  if (out.getDimSize(0) != lhs.getDimSize(0) ||
      out.getDimSize(1) != rhs.getDimSize(1))
    return emitOpError("result must be ")
           << lhs.getDimSize(0) << "x" << rhs.getDimSize(1);
  return success();
}

LogicalResult TransposeOp::verify() {
  auto in = getInput().getType();
  auto out = getOutput().getType();
  ArrayRef<int64_t> perm = getPermutation();
  if (static_cast<int64_t>(perm.size()) != in.getRank())
    return emitOpError("permutation has ")
           << perm.size() << " entries for a rank-" << in.getRank()
           << " input";
  SmallVector<bool> seen(perm.size(), false);
  for (int64_t p : perm) {
    if (p < 0 || p >= in.getRank() || seen[p])
      return emitOpError("permutation is not a permutation of [0, ")
             << in.getRank() << ")";
    seen[p] = true;
  }
  for (auto [i, p] : llvm::enumerate(perm))
    if (out.getDimSize(i) != in.getDimSize(p))
      return emitOpError("result dim ")
             << i << " is " << out.getDimSize(i) << ", expected input dim "
             << p << " (" << in.getDimSize(p) << ")";
  return success();
}

LogicalResult FusedElementwiseOp::verify() {
  auto outType = getOutput().getType();
  for (Value in : getInputs())
    if (in.getType() != outType)
      return emitOpError("every input must have the result type ") << outType;

  Block &body = getBody().front();
  if (body.getNumArguments() != getInputs().size())
    return emitOpError("body takes ")
           << body.getNumArguments() << " arguments for "
           << getInputs().size() << " inputs";
  auto f32 = Float32Type::get(getContext());
  for (BlockArgument arg : body.getArguments())
    if (arg.getType() != f32)
      return emitOpError("body arguments must be f32 scalars");
  if (!isa<YieldOp>(body.getTerminator()))
    return emitOpError("body must end in topt.yield");
  return success();
}
