// A non-splat result above the budget (2^18 elements) is not folded: folding
// would put 4 MiB of constant data into the binary to save one loop.
// The input is generated because a literal would be millions of characters.
// RUN: %python %S/Inputs/gen_big_constant.py 1024 1024 | topt-opt -topt-constant-fold | FileCheck %s
// RUN: %python %S/Inputs/gen_big_constant.py 512 512 | topt-opt -topt-constant-fold | FileCheck %s --check-prefix=UNDER

// CHECK: topt.add
// UNDER-NOT: topt.add
