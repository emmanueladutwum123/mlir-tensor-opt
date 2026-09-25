#include "TensorOpt/TensorOptDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::topt;

#define GET_OP_CLASSES
#include "TensorOpt/TensorOptOps.cpp.inc"
