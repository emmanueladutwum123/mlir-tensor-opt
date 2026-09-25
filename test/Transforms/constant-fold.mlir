// RUN: topt-opt %s -topt-constant-fold -split-input-file | FileCheck %s

// A chain of constant elementwise ops collapses into one constant, and the
// intermediate constants are erased. (a + b) * b + a = [3,5]*[2,3] + [1,2].
// CHECK-LABEL: func.func @elementwise_chain
// CHECK-NEXT:    %[[C:.*]] = topt.constant dense<[7.000000e+00, 1.700000e+01]> : tensor<2xf32>
// CHECK-NEXT:    return %[[C]]
func.func @elementwise_chain() -> tensor<2xf32> {
  %a = topt.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %b = topt.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %s = topt.add %a, %b : tensor<2xf32>
  %p = topt.mul %s, %b : tensor<2xf32>
  %r = topt.add %p, %a : tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// Folding is IEEE single precision with one rounding per op: 16777216 + 1
// rounds back to 16777216 = 2^24 in f32 (the answer in f64 would be
// 16777217). MLIR prints the result as its bit pattern: 0x4B800000 is 2^24.
// CHECK-LABEL: func.func @f32_rounding
// CHECK: topt.constant dense<0x4B800000> : tensor<1xf32>
func.func @f32_rounding() -> tensor<1xf32> {
  %a = topt.constant dense<16777216.0> : tensor<1xf32>
  %b = topt.constant dense<1.0> : tensor<1xf32>
  %s = topt.add %a, %b : tensor<1xf32>
  return %s : tensor<1xf32>
}

// -----

// A non-splat transpose really permutes: [[1,2,3],[4,5,6]]^T.
// CHECK-LABEL: func.func @transpose_2d
// CHECK: topt.constant dense<{{\[}}[1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00], [3.000000e+00, 6.000000e+00]]> : tensor<3x2xf32>
// CHECK-NOT: topt.transpose
func.func @transpose_2d() -> tensor<3x2xf32> {
  %a = topt.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %t = topt.transpose %a, [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  return %t : tensor<3x2xf32>
}

// -----

// Rank 3, permutation [2, 0, 1]: out[i][j][k] = in[j][k][i].
// in = [[[0,1]], [[2,3]]] has shape 2x1x2, out has shape 2x2x1.
// CHECK-LABEL: func.func @transpose_3d
// CHECK: topt.constant dense<{{\[}}{{\[}}[0.000000e+00], [2.000000e+00]], {{\[}}[1.000000e+00], [3.000000e+00]]]> : tensor<2x2x1xf32>
func.func @transpose_3d() -> tensor<2x2x1xf32> {
  %a = topt.constant dense<[[[0.0, 1.0]], [[2.0, 3.0]]]> : tensor<2x1x2xf32>
  %t = topt.transpose %a, [2, 0, 1] : tensor<2x1x2xf32> -> tensor<2x2x1xf32>
  return %t : tensor<2x2x1xf32>
}

// -----

// Matmul folds: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]].
// CHECK-LABEL: func.func @matmul
// CHECK: topt.constant dense<{{\[}}[1.900000e+01, 2.200000e+01], [4.300000e+01, 5.000000e+01]]> : tensor<2x2xf32>
// CHECK-NOT: topt.matmul
func.func @matmul() -> tensor<2x2xf32> {
  %a = topt.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b = topt.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %c = topt.matmul %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

// -----

// Partially constant expressions fold the constant subtree and leave the rest.
// CHECK-LABEL: func.func @partial
// CHECK:      %[[C:.*]] = topt.constant dense<3.000000e+00> : tensor<4xf32>
// CHECK-NEXT: %[[R:.*]] = topt.mul %arg0, %[[C]] : tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @partial(%x: tensor<4xf32>) -> tensor<4xf32> {
  %a = topt.constant dense<1.0> : tensor<4xf32>
  %b = topt.constant dense<2.0> : tensor<4xf32>
  %s = topt.add %a, %b : tensor<4xf32>
  %r = topt.mul %x, %s : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// This pass evaluates constants and nothing else: an identity it could
// prove is left for -topt-simplify.
// CHECK-LABEL: func.func @does_not_simplify
// CHECK-COUNT-2: topt.transpose
func.func @does_not_simplify(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %t = topt.transpose %x, [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  %u = topt.transpose %t, [1, 0] : tensor<3x2xf32> -> tensor<2x3xf32>
  return %u : tensor<2x3xf32>
}

// -----

// A splat folds at any size: its result costs one scalar.
// CHECK-LABEL: func.func @big_splat
// CHECK: topt.constant dense<3.000000e+00> : tensor<1024x1024xf32>
// CHECK-NOT: topt.add
func.func @big_splat() -> tensor<1024x1024xf32> {
  %a = topt.constant dense<1.0> : tensor<1024x1024xf32>
  %b = topt.constant dense<2.0> : tensor<1024x1024xf32>
  %s = topt.add %a, %b : tensor<1024x1024xf32>
  return %s : tensor<1024x1024xf32>
}
