import os

import lit.formats

config.name = "TOPT"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.excludes = ["lit.cfg.py", "lit.site.cfg.py.in", "CMakeLists.txt"]

# topt-opt from this build; FileCheck, not, mlir-runner, mlir-translate from
# the LLVM installation that the project was configured against.
config.environment["PATH"] = os.pathsep.join(
    [config.topt_tools_dir, config.llvm_tools_dir, config.environment.get("PATH", "")]
)

runner_libs = ",".join(
    os.path.join(config.mlir_lib_dir, f"lib{name}{config.shlib_ext}")
    for name in ("mlir_runner_utils", "mlir_c_runner_utils")
)
config.substitutions.append(("%mlir_runner_libs", runner_libs))
config.excludes.append("Inputs")
import sys; config.substitutions.append(("%python", sys.executable))
