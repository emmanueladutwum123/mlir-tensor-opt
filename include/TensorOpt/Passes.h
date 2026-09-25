#ifndef TENSOROPT_PASSES_H
#define TENSOROPT_PASSES_H

#include "TensorOpt/TensorOptDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::topt {

#define GEN_PASS_DECL
#include "TensorOpt/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TensorOpt/Passes.h.inc"

/// Register -topt-lower-to-llvm: convert-topt-to-linalg followed by the
/// upstream bufferization, loop and LLVM lowering passes.
void registerLowerToLLVMPipeline();

} // namespace mlir::topt

#endif // TENSOROPT_PASSES_H
