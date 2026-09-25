// Each verifier rule rejects the IR it should, with the message it should.
// RUN: topt-opt %s -split-input-file -verify-diagnostics

func.func @matmul_inner_dims(%a: tensor<2x3xf32>, %b: tensor<4x5xf32>) {
  // expected-error @+1 {{contraction dimensions differ: lhs has 3 columns, rhs has 4 rows}}
  %0 = topt.matmul %a, %b : tensor<2x3xf32>, tensor<4x5xf32> -> tensor<2x5xf32>
  return
}

// -----

func.func @matmul_result_shape(%a: tensor<2x3xf32>, %b: tensor<3x5xf32>) {
  // expected-error @+1 {{result must be 2x5}}
  %0 = topt.matmul %a, %b : tensor<2x3xf32>, tensor<3x5xf32> -> tensor<5x2xf32>
  return
}

// -----

func.func @matmul_rank(%a: tensor<2x3x4xf32>, %b: tensor<4x5xf32>) {
  // expected-error @+1 {{expects rank-2 operands and result}}
  %0 = topt.matmul %a, %b : tensor<2x3x4xf32>, tensor<4x5xf32> -> tensor<2x5xf32>
  return
}

// -----

func.func @transpose_not_a_permutation(%a: tensor<2x3xf32>) {
  // expected-error @+1 {{permutation is not a permutation of [0, 2)}}
  %0 = topt.transpose %a, [0, 0] : tensor<2x3xf32> -> tensor<2x3xf32>
  return
}

// -----

func.func @transpose_wrong_length(%a: tensor<2x3xf32>) {
  // expected-error @+1 {{permutation has 3 entries for a rank-2 input}}
  %0 = topt.transpose %a, [0, 1, 2] : tensor<2x3xf32> -> tensor<2x3xf32>
  return
}

// -----

func.func @transpose_result_shape(%a: tensor<2x3xf32>) {
  // expected-error @+1 {{result dim 0 is 2, expected input dim 1 (3)}}
  %0 = topt.transpose %a, [1, 0] : tensor<2x3xf32> -> tensor<2x3xf32>
  return
}

// -----

// expected-note @+1 {{prior use here}}
func.func @add_mismatched(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) {
  // expected-error @+1 {{use of value '%b' expects different type than prior uses}}
  %0 = topt.add %a, %b : tensor<2x3xf32>
  return
}

// -----

func.func @dynamic_shape(%a: tensor<?xf32>) {
  // expected-error @+1 {{must be statically shaped tensor of 32-bit float values, but got 'tensor<?xf32>'}}
  %0 = topt.add %a, %a : tensor<?xf32>
  return
}

// -----

func.func @wrong_element_type(%a: tensor<4xi32>) {
  // expected-error @+1 {{must be statically shaped tensor of 32-bit float values, but got 'tensor<4xi32>'}}
  %0 = topt.add %a, %a : tensor<4xi32>
  return
}

// -----

func.func @fused_arity(%a: tensor<4xf32>) {
  // expected-error @+1 {{body takes 2 arguments for 1 inputs}}
  %0 = topt.fused_elementwise %a : tensor<4xf32> -> tensor<4xf32> {
  ^bb0(%x: f32, %y: f32):
    topt.yield %x : f32
  }
  return
}
