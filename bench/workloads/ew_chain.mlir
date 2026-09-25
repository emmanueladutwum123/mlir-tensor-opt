// Target: fusion. Eight elementwise ops over 2048x2048 (16 MiB per tensor,
// larger than the M2's 16 MiB L2 once there are several of them). Unfused,
// every op is a full pass over memory and seven 16 MiB intermediates are
// allocated. Fused, it is one pass that reads six inputs and writes once.
func.func @kernel(%a: tensor<2048x2048xf32>, %b: tensor<2048x2048xf32>,
                  %c: tensor<2048x2048xf32>, %d: tensor<2048x2048xf32>,
                  %e: tensor<2048x2048xf32>, %f: tensor<2048x2048xf32>)
    -> tensor<2048x2048xf32> attributes {llvm.emit_c_interface} {
  %0 = topt.mul %a, %b : tensor<2048x2048xf32>
  %1 = topt.add %0, %c : tensor<2048x2048xf32>
  %2 = topt.mul %1, %d : tensor<2048x2048xf32>
  %3 = topt.add %2, %e : tensor<2048x2048xf32>
  %4 = topt.mul %3, %f : tensor<2048x2048xf32>
  %5 = topt.add %4, %a : tensor<2048x2048xf32>
  %6 = topt.mul %5, %b : tensor<2048x2048xf32>
  %7 = topt.add %6, %c : tensor<2048x2048xf32>
  return %7 : tensor<2048x2048xf32>
}
