// The whole pipeline ends in the LLVM dialect: no topt, linalg, tensor,
// memref, affine or scf op survives, and mlir-translate accepts the result.
// RUN: topt-opt %s -topt-lower-to-llvm | FileCheck %s
// RUN: topt-opt %s -topt-lower-to-llvm | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR

// CHECK-NOT: topt.
// CHECK-NOT: linalg.
// CHECK-NOT: tensor.
// CHECK-NOT: memref.
// CHECK-NOT: affine.
// CHECK-NOT: scf.
// CHECK: llvm.func @f(
// CHECK: llvm.call @malloc
// CHECK: llvm.func @_mlir_ciface_f(

// LLVMIR: define { ptr, ptr, i64, [2 x i64], [2 x i64] } @f(
// LLVMIR: fmul float
// LLVMIR: fadd float
// LLVMIR: define void @_mlir_ciface_f(

func.func @f(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32>
    attributes {llvm.emit_c_interface} {
  %0 = topt.matmul %a, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  %1 = topt.transpose %0, [1, 0] : tensor<4x4xf32> -> tensor<4x4xf32>
  %2 = topt.add %1, %a : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
