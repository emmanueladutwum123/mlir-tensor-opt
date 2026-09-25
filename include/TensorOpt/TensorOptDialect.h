#ifndef TENSOROPT_TENSOROPTDIALECT_H
#define TENSOROPT_TENSOROPTDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TensorOpt/TensorOptOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TensorOpt/TensorOptOps.h.inc"

#endif // TENSOROPT_TENSOROPTDIALECT_H
