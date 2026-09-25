add_library(TensorOpt STATIC
  TensorOptDialect.cpp
  TensorOptOps.cpp
  ConstantFold.cpp
  Simplify.cpp
  Fusion.cpp
  LowerToLinalg.cpp
  Pipeline.cpp
)
add_dependencies(TensorOpt TensorOptOpsIncGen TensorOptPassesIncGen)
# Homebrew's MLIR package sets MLIR_LINK_MLIR_DYLIB, and this then links the
# one shared libMLIR instead of the static archives, which alongside libLLVM
# would risk two copies of LLVM's global option registry.
mlir_target_link_libraries(TensorOpt PUBLIC MLIRIR MLIRSupport MLIRPass
  MLIRTransformUtils MLIRFuncDialect MLIRArithDialect
  MLIRLinalgDialect MLIRTensorDialect)
# libMLIR does not re-export LLVM's Support library; that lives in libLLVM.
# (Where MLIR is linked statically, its archives pull LLVM in themselves.)
if(MLIR_LINK_MLIR_DYLIB AND LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(TensorOpt PUBLIC LLVM)
endif()
