#include "TensorOpt/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::topt {

// Everything after convert-topt-to-linalg is upstream MLIR, unchanged.
//
// Rejected: convert-linalg-to-loops (scf). The affine route keeps loop
// bounds and subscripts analysable, which is the dialect you would tile or
// interchange in next; lower-affine turns it into scf/cf anyway.
static constexpr llvm::StringLiteral kUpstreamLowering =
    "one-shot-bufferize{bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map},"
    "buffer-deallocation-pipeline,"
    "func.func(convert-linalg-to-affine-loops),"
    "lower-affine,"
    "convert-scf-to-cf,"
    "expand-strided-metadata,"
    "finalize-memref-to-llvm,"
    "convert-arith-to-llvm,"
    "convert-index-to-llvm,"
    "convert-cf-to-llvm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts";

void registerLowerToLLVMPipeline() {
  PassPipelineRegistration<>(
      "topt-lower-to-llvm",
      "Lower topt to the LLVM dialect via linalg, bufferization and affine",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createConvertToptToLinalg());
        if (failed(parsePassPipeline(kUpstreamLowering, pm)))
          llvm::report_fatal_error("topt-lower-to-llvm: bad pipeline string");
      });
}

} // namespace mlir::topt
