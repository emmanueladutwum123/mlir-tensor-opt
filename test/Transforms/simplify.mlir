// RUN: topt-opt %s -topt-simplify -split-input-file | FileCheck %s

// The headline rewrite: both transposes vanish.
// CHECK-LABEL: func.func @transpose_transpose
// CHECK-NEXT:    return %arg0
func.func @transpose_transpose(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %t = topt.transpose %x, [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  %u = topt.transpose %t, [1, 0] : tensor<3x2xf32> -> tensor<2x3xf32>
  return %u : tensor<2x3xf32>
}

// -----

// Rank 3: two rotations by [1,2,0] compose to [2,0,1] (not the identity),
// so the pair becomes ONE transpose rather than disappearing.
// CHECK-LABEL: func.func @compose_rank3
// CHECK-NEXT:    %[[T:.*]] = topt.transpose %arg0, [2, 0, 1] : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
// CHECK-NEXT:    return %[[T]]
func.func @compose_rank3(%x: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> {
  %t = topt.transpose %x, [1, 2, 0] : tensor<2x3x5xf32> -> tensor<3x5x2xf32>
  %u = topt.transpose %t, [1, 2, 0] : tensor<3x5x2xf32> -> tensor<5x2x3xf32>
  return %u : tensor<5x2x3xf32>
}

// -----

// Three rotations by [1,2,0] are the identity.
// CHECK-LABEL: func.func @three_rotations
// CHECK-NEXT:    return %arg0
func.func @three_rotations(%x: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
  %a = topt.transpose %x, [1, 2, 0] : tensor<2x3x5xf32> -> tensor<3x5x2xf32>
  %b = topt.transpose %a, [1, 2, 0] : tensor<3x5x2xf32> -> tensor<5x2x3xf32>
  %c = topt.transpose %b, [1, 2, 0] : tensor<5x2x3xf32> -> tensor<2x3x5xf32>
  return %c : tensor<2x3x5xf32>
}

// -----

// The inner transpose has another user, so it must survive; only the outer
// one is removed.
// CHECK-LABEL: func.func @shared_inner
// CHECK-NEXT:    %[[T:.*]] = topt.transpose %arg0, [1, 0]
// CHECK-NEXT:    return %arg0, %[[T]]
func.func @shared_inner(%x: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<3x2xf32>) {
  %t = topt.transpose %x, [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  %u = topt.transpose %t, [1, 0] : tensor<3x2xf32> -> tensor<2x3xf32>
  return %u, %t : tensor<2x3xf32>, tensor<3x2xf32>
}

// -----

// CHECK-LABEL: func.func @identity_transpose
// CHECK-NEXT:    return %arg0
func.func @identity_transpose(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %t = topt.transpose %x, [0, 1] : tensor<2x3xf32> -> tensor<2x3xf32>
  return %t : tensor<2x3xf32>
}

// -----

// x * 1 -> x, with the constant on either side.
// CHECK-LABEL: func.func @mul_one
// CHECK-NOT:     topt.mul
// CHECK:         return %arg0, %arg0
func.func @mul_one(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %one = topt.constant dense<1.0> : tensor<4xf32>
  %a = topt.mul %x, %one : tensor<4xf32>
  %b = topt.mul %one, %x : tensor<4xf32>
  return %a, %b : tensor<4xf32>, tensor<4xf32>
}

// -----

// x + (-0.0) -> x, with the constant on either side.
// CHECK-LABEL: func.func @add_negative_zero
// CHECK-NOT:     topt.add
// CHECK:         return %arg0, %arg0
func.func @add_negative_zero(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %nz = topt.constant dense<-0.0> : tensor<4xf32>
  %a = topt.add %x, %nz : tensor<4xf32>
  %b = topt.add %nz, %x : tensor<4xf32>
  return %a, %b : tensor<4xf32>, tensor<4xf32>
}

// -----

// NOT rewritten: x + (+0.0) is -0.0 + 0.0 = +0.0 for x = -0.0, and
// x * 0.0 is NaN for x = inf or NaN and -0.0 for negative x.
// CHECK-LABEL: func.func @unsound_identities_are_kept
// CHECK:         topt.add %arg0
// CHECK:         topt.mul %arg0
func.func @unsound_identities_are_kept(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %z = topt.constant dense<0.0> : tensor<4xf32>
  %a = topt.add %x, %z : tensor<4xf32>
  %b = topt.mul %x, %z : tensor<4xf32>
  return %a, %b : tensor<4xf32>, tensor<4xf32>
}

// -----

// A non-splat constant that happens to contain a 1 is not the identity.
// CHECK-LABEL: func.func @non_splat_not_identity
// CHECK:         topt.mul
func.func @non_splat_not_identity(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = topt.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %a = topt.mul %x, %c : tensor<2xf32>
  return %a : tensor<2xf32>
}

// -----

// This pass does not evaluate constants; that is -topt-constant-fold's job.
// CHECK-LABEL: func.func @does_not_fold
// CHECK:         topt.add
func.func @does_not_fold() -> tensor<2xf32> {
  %a = topt.constant dense<1.0> : tensor<2xf32>
  %s = topt.add %a, %a : tensor<2xf32>
  return %s : tensor<2xf32>
}
