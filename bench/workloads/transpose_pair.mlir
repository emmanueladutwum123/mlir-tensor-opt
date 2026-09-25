// Target: simplification. A layout round-trip (transpose to column-major
// and back, as happens when two library conventions meet) before one add.
// Each transpose is a full strided copy of 16 MiB; simplify deletes both.
func.func @kernel(%a: tensor<2048x2048xf32>, %b: tensor<2048x2048xf32>)
    -> tensor<2048x2048xf32> attributes {llvm.emit_c_interface} {
  %t = topt.transpose %a, [1, 0] : tensor<2048x2048xf32> -> tensor<2048x2048xf32>
  %u = topt.transpose %t, [1, 0] : tensor<2048x2048xf32> -> tensor<2048x2048xf32>
  %r = topt.add %u, %b : tensor<2048x2048xf32>
  return %r : tensor<2048x2048xf32>
}
