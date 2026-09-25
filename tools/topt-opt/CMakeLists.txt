add_executable(topt-opt topt-opt.cpp)
target_link_libraries(topt-opt PRIVATE TensorOpt)
mlir_target_link_libraries(topt-opt PRIVATE MLIROptLib MLIRRegisterAllDialects
  MLIRRegisterAllExtensions MLIRRegisterAllPasses)
