#ifndef TENSOROPT_PASSES_H
#define TENSOROPT_PASSES_H

#include "TensorOpt/TensorOptDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::topt {

#define GEN_PASS_DECL
#include "TensorOpt/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TensorOpt/Passes.h.inc"

} // namespace mlir::topt

#endif // TENSOROPT_PASSES_H
