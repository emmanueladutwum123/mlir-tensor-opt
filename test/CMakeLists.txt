configure_file(lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py @ONLY)

find_program(LIT_EXECUTABLE NAMES lit llvm-lit REQUIRED)
add_custom_target(check-topt
  COMMAND ${LIT_EXECUTABLE} -sv ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS topt-opt
  USES_TERMINAL
  COMMENT "Running topt lit tests")
