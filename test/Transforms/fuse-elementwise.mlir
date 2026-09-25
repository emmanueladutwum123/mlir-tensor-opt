// RUN: topt-opt %s -topt-fuse-elementwise -split-input-file | FileCheck %s

// A linear chain of four ops becomes one op; each scalar op appears once,
// in the original order.
// CHECK-LABEL: func.func @chain
// CHECK:         %[[F:.*]] = topt.fused_elementwise %arg0, %arg1, %arg2, %arg3, %arg4 : {{.*}} -> tensor<16xf32> {
// CHECK-NEXT:    ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32, %[[D:.*]]: f32, %[[E:.*]]: f32):
// CHECK-NEXT:      %[[S0:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK-NEXT:      %[[S1:.*]] = arith.addf %[[S0]], %[[C]] : f32
// CHECK-NEXT:      %[[S2:.*]] = arith.mulf %[[S1]], %[[D]] : f32
// CHECK-NEXT:      %[[S3:.*]] = arith.addf %[[S2]], %[[E]] : f32
// CHECK-NEXT:      topt.yield %[[S3]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[F]]
func.func @chain(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<16xf32>,
                 %d: tensor<16xf32>, %e: tensor<16xf32>) -> tensor<16xf32> {
  %0 = topt.mul %a, %b : tensor<16xf32>
  %1 = topt.add %0, %c : tensor<16xf32>
  %2 = topt.mul %1, %d : tensor<16xf32>
  %3 = topt.add %2, %e : tensor<16xf32>
  return %3 : tensor<16xf32>
}

// -----

// A tree: (a + b) * (c + d). Both producers have one user, so all three ops
// fuse.
// CHECK-LABEL: func.func @tree
// CHECK-COUNT-1: topt.fused_elementwise
// CHECK-NOT:     topt.add
// CHECK-NOT:     topt.mul
func.func @tree(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>,
                %d: tensor<4xf32>) -> tensor<4xf32> {
  %0 = topt.add %a, %b : tensor<4xf32>
  %1 = topt.add %c, %d : tensor<4xf32>
  %2 = topt.mul %0, %1 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// A tensor read twice becomes one input and one block argument.
// CHECK-LABEL: func.func @dedup
// CHECK:         topt.fused_elementwise %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32> {
// CHECK-NEXT:    ^bb0(%[[X:.*]]: f32, %[[Y:.*]]: f32):
// CHECK-NEXT:      %[[S:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:      arith.mulf %[[S]], %[[X]]
func.func @dedup(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %0 = topt.add %x, %y : tensor<4xf32>
  %1 = topt.mul %0, %x : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// The consumer reads the producer twice: still its only user, so it fuses,
// and the producer's scalar is computed once and used twice.
// CHECK-LABEL: func.func @square
// CHECK:         ^bb0(%[[X:.*]]: f32, %[[Y:.*]]: f32):
// CHECK-NEXT:      %[[S:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:      arith.mulf %[[S]], %[[S]]
func.func @square(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %0 = topt.add %x, %y : tensor<4xf32>
  %1 = topt.mul %0, %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// The producer has a second user, so fusing would mean computing it twice.
// Nothing fuses.
// CHECK-LABEL: func.func @multi_use_producer
// CHECK-NOT:     topt.fused_elementwise
// CHECK:         %[[S:.*]] = topt.add
// CHECK:         topt.mul %[[S]]
// CHECK:         return
func.func @multi_use_producer(%x: tensor<4xf32>, %y: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = topt.add %x, %y : tensor<4xf32>
  %1 = topt.mul %0, %y : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// Fusion stops at non-elementwise ops: a matmul in the middle splits the
// chain into one fused op before it and one after it.
// CHECK-LABEL: func.func @stops_at_matmul
// CHECK:         %[[F0:.*]] = topt.fused_elementwise
// CHECK:         %[[M:.*]] = topt.matmul %[[F0]]
// CHECK:         topt.fused_elementwise %[[M]]
func.func @stops_at_matmul(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = topt.add %a, %b : tensor<4x4xf32>
  %1 = topt.mul %0, %b : tensor<4x4xf32>
  %2 = topt.matmul %1, %a : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  %3 = topt.add %2, %a : tensor<4x4xf32>
  %4 = topt.mul %3, %b : tensor<4x4xf32>
  return %4 : tensor<4x4xf32>
}

// -----

// A single op has no producer to fuse with and is left alone.
// CHECK-LABEL: func.func @single
// CHECK-NOT:     topt.fused_elementwise
// CHECK:         topt.add
func.func @single(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %0 = topt.add %x, %y : tensor<4xf32>
  return %0 : tensor<4xf32>
}
