// transpose_pair at 2000x2000 instead of 2048x2048. 2048 is a power of two,
// so a naive transpose walks memory with an 8 KiB stride that thrashes
// cache sets and the TLB. Comparing the two sizes separates what
// -topt-simplify saves from what that pathology inflates.
func.func @kernel(%a: tensor<2000x2000xf32>, %b: tensor<2000x2000xf32>)
    -> tensor<2000x2000xf32> attributes {llvm.emit_c_interface} {
  %t = topt.transpose %a, [1, 0] : tensor<2000x2000xf32> -> tensor<2000x2000xf32>
  %u = topt.transpose %t, [1, 0] : tensor<2000x2000xf32> -> tensor<2000x2000xf32>
  %r = topt.add %u, %b : tensor<2000x2000xf32>
  return %r : tensor<2000x2000xf32>
}
