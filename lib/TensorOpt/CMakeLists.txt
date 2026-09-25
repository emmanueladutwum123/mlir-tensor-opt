add_library(TensorOpt STATIC
  TensorOptDialect.cpp
  TensorOptOps.cpp
  ConstantFold.cpp
  Simplify.cpp
  Fusion.cpp
)
add_dependencies(TensorOpt TensorOptOpsIncGen TensorOptPassesIncGen)
# MLIR_LINK_MLIR_DYLIB is set by Homebrew's build: link the one shared libMLIR
# rather than the static archives, which would duplicate LLVM's global
# command-line option registry and abort at startup.
mlir_target_link_libraries(TensorOpt PUBLIC MLIRIR MLIRSupport MLIRPass
  MLIRTransformUtils MLIRFuncDialect MLIRArithDialect)
# libMLIR does not re-export LLVM's Support library; that lives in libLLVM.
target_link_libraries(TensorOpt PUBLIC LLVM)
