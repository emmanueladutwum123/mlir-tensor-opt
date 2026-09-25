// RUN: topt-opt %s -convert-topt-to-linalg -split-input-file | FileCheck %s

// add -> one linalg.generic with identity maps and an addf body.
// CHECK: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @add
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x3xf32>
// CHECK:         linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:      ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%[[E]] : tensor<2x3xf32>)
// CHECK:           arith.addf
// CHECK:           linalg.yield
// CHECK-NOT:     topt.
func.func @add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = topt.add %a, %b : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// A splat operand becomes a scalar in the body, not a tensor input, and the
// tensor constant it came from is erased.
// CHECK-LABEL: func.func @mul_splat
// CHECK-NOT:     arith.constant dense
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<4xf32>)
// CHECK:           %[[C:.*]] = arith.constant 3.000000e+00 : f32
// CHECK:           arith.mulf %{{.*}}, %[[C]]
func.func @mul_splat(%a: tensor<4xf32>) -> tensor<4xf32> {
  %c = topt.constant dense<3.0> : tensor<4xf32>
  %0 = topt.mul %a, %c : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A non-splat constant stays a tensor.
// CHECK-LABEL: func.func @non_splat_constant
// CHECK:         %[[C:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0, %[[C]] : tensor<2xf32>, tensor<2xf32>)
func.func @non_splat_constant(%a: tensor<2xf32>) -> tensor<2xf32> {
  %c = topt.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = topt.add %a, %c : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @transpose
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<5x2x3xf32>
// CHECK:         linalg.transpose ins(%arg0 : tensor<2x3x5xf32>) outs(%[[E]] : tensor<5x2x3xf32>) permutation = [2, 0, 1]
func.func @transpose(%a: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> {
  %0 = topt.transpose %a, [2, 0, 1] : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
  return %0 : tensor<5x2x3xf32>
}

// -----

// matmul accumulates, so the output is zero-filled first.
// CHECK-LABEL: func.func @matmul
// CHECK:         %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:         %[[F:.*]] = linalg.fill ins(%[[Z]] : f32) outs(%[[E]] : tensor<2x4xf32>)
// CHECK:         linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%[[F]] : tensor<2x4xf32>)
func.func @matmul(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %0 = topt.matmul %a, %b : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// fused_elementwise -> ONE generic whose body is the fused body.
// CHECK-LABEL: func.func @fused
// CHECK-COUNT-1: linalg.generic
// CHECK:           %[[S:.*]] = arith.addf %[[X:.*]], %[[Y:.*]] : f32
// CHECK-NEXT:      %[[M:.*]] = arith.mulf %[[S]], %[[X]] : f32
// CHECK-NEXT:      linalg.yield %[[M]] : f32
// CHECK-NOT:     linalg.generic
func.func @fused(%x: tensor<8xf32>, %y: tensor<8xf32>) -> tensor<8xf32> {
  %r = topt.fused_elementwise %x, %y : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32> {
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    %m = arith.mulf %s, %a : f32
    topt.yield %m : f32
  }
  return %r : tensor<8xf32>
}
