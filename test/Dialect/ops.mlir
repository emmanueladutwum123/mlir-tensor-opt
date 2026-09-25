// Every op parses, verifies, prints, and re-parses to the same thing.
// RUN: topt-opt %s | topt-opt | FileCheck %s

// CHECK-LABEL: func.func @all_ops
func.func @all_ops(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>,
                   %m: tensor<3x4xf32>) -> tensor<4x2xf32> {
  // CHECK: topt.constant dense<1.000000e+00> : tensor<2x3xf32>
  %c = topt.constant dense<1.0> : tensor<2x3xf32>
  // CHECK: topt.add %{{.*}}, %{{.*}} : tensor<2x3xf32>
  %s = topt.add %a, %c : tensor<2x3xf32>
  // CHECK: topt.mul %{{.*}}, %{{.*}} : tensor<2x3xf32>
  %p = topt.mul %s, %b : tensor<2x3xf32>
  // CHECK: topt.matmul %{{.*}}, %{{.*}} : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
  %mm = topt.matmul %p, %m : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
  // CHECK: topt.transpose %{{.*}}, [1, 0] : tensor<2x4xf32> -> tensor<4x2xf32>
  %t = topt.transpose %mm, [1, 0] : tensor<2x4xf32> -> tensor<4x2xf32>
  return %t : tensor<4x2xf32>
}

// CHECK-LABEL: func.func @fused
func.func @fused(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: topt.fused_elementwise %{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32> {
  // CHECK-NEXT: ^bb0(%[[X:.*]]: f32, %[[Y:.*]]: f32):
  // CHECK-NEXT:   %[[S:.*]] = arith.addf %[[X]], %[[Y]] : f32
  // CHECK-NEXT:   topt.yield %[[S]] : f32
  %r = topt.fused_elementwise %a, %b : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32> {
  ^bb0(%x: f32, %y: f32):
    %s = arith.addf %x, %y : f32
    topt.yield %s : f32
  }
  return %r : tensor<8xf32>
}

// CHECK-LABEL: func.func @rank3_transpose
func.func @rank3_transpose(%a: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> {
  // CHECK: topt.transpose %{{.*}}, [2, 0, 1]
  %t = topt.transpose %a, [2, 0, 1] : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
  return %t : tensor<5x2x3xf32>
}
