// topt-opt: mlir-opt with the topt dialect and passes registered.

#include "TensorOpt/TensorOptDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::topt::TensorOptDialect>();
  // Upstream dialects plus their external interface models; one-shot
  // bufferization and convert-to-llvm find their per-dialect hooks this way.
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "topt optimizer driver\n", registry));
}
