// Run the same program, compiled to native code and JIT-executed, under
// every pass configuration. Each must print the same numbers: this is the
// test that the passes preserve meaning, not just that they change the IR.
//
// DEFINE: %{run} = mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_libs
// RUN: topt-opt %s -topt-lower-to-llvm | %{run} | FileCheck %s
// RUN: topt-opt %s -topt-constant-fold -topt-lower-to-llvm | %{run} | FileCheck %s
// RUN: topt-opt %s -topt-simplify -topt-lower-to-llvm | %{run} | FileCheck %s
// RUN: topt-opt %s -topt-fuse-elementwise -topt-lower-to-llvm | %{run} | FileCheck %s
// RUN: topt-opt %s -topt-constant-fold -topt-simplify -topt-fuse-elementwise -topt-lower-to-llvm | %{run} | FileCheck %s
//
// And the fully optimised @compute really is smaller: one transpose pair,
// the x*1 and the x+(-0) are gone, and the add and mul left fuse: 11 ops -> 2.
// RUN: topt-opt %s -topt-constant-fold -topt-simplify -topt-fuse-elementwise | FileCheck %s --check-prefix=OPT

// x = [[1,2,3],[4,5,6]], y = [[1,1,1],[2,2,2]], w = [[1,0],[0,1],[1,1]]
// b = x + y = [[2,3,4],[6,7,8]];  c = b * y = [[2,3,4],[12,14,16]]
// c @ w = [[2+4, 3+4], [12+16, 14+16]] = [[6, 7], [28, 30]]
// CHECK:      sizes = [2, 2]
// CHECK-NEXT: {{\[\[}}6,{{ +}}7],
// CHECK-NEXT: [28,{{ +}}30]]

// OPT-LABEL: func.func @compute
// OPT-NOT:   topt.transpose
// OPT-NOT:   topt.constant
// OPT:       topt.fused_elementwise %arg0, %arg1
// OPT-NEXT:  ^bb0
// OPT-NEXT:    arith.addf
// OPT-NEXT:    arith.mulf
// OPT-NEXT:    topt.yield
// OPT-NEXT:  }
// OPT-NEXT:  topt.matmul
// OPT-NEXT:  return

func.func @compute(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>,
                   %w: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %half = topt.constant dense<0.5> : tensor<2x3xf32>
  %two = topt.constant dense<2.0> : tensor<2x3xf32>
  %nz = topt.constant dense<-0.0> : tensor<2x3xf32>
  %scale = topt.mul %half, %two : tensor<2x3xf32>
  %t0 = topt.transpose %x, [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  %t1 = topt.transpose %t0, [1, 0] : tensor<3x2xf32> -> tensor<2x3xf32>
  %a = topt.mul %t1, %scale : tensor<2x3xf32>
  %b = topt.add %a, %y : tensor<2x3xf32>
  %c = topt.mul %b, %y : tensor<2x3xf32>
  %d = topt.add %c, %nz : tensor<2x3xf32>
  %r = topt.matmul %d, %w : tensor<2x3xf32>, tensor<3x2xf32> -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

func.func @main() {
  %x = topt.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %y = topt.constant dense<[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]> : tensor<2x3xf32>
  %w = topt.constant dense<[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]> : tensor<3x2xf32>
  %r = call @compute(%x, %y, %w) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  %u = tensor.cast %r : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(tensor<*xf32>) attributes {llvm.emit_c_interface}
