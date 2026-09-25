// The scale 0.5 * 2.0 is 1.0, but only after it is folded. Neither pass
// alone removes the multiply; folding first and then simplifying does.
// This is why the pipeline order is fold -> simplify -> fuse.
//
// RUN: topt-opt %s -topt-simplify | FileCheck %s --check-prefix=SIMPLIFY
// RUN: topt-opt %s -topt-constant-fold | FileCheck %s --check-prefix=FOLD
// RUN: topt-opt %s -topt-constant-fold -topt-simplify | FileCheck %s --check-prefix=BOTH

// SIMPLIFY: topt.mul %arg0
// FOLD:     topt.constant dense<1.000000e+00>
// FOLD:     topt.mul %arg0
// BOTH:     func.func @scale
// BOTH-NOT: topt.mul
// BOTH:     return %arg0
func.func @scale(%x: tensor<8xf32>) -> tensor<8xf32> {
  %half = topt.constant dense<0.5> : tensor<8xf32>
  %two = topt.constant dense<2.0> : tensor<8xf32>
  %s = topt.mul %half, %two : tensor<8xf32>
  %y = topt.mul %x, %s : tensor<8xf32>
  return %y : tensor<8xf32>
}
