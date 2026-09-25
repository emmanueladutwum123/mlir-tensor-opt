// All three passes together, on something shaped like a layer: a matmul,
// a layout round-trip, a scale that is 1.0 only after folding (0.5 * 2.0),
// a bias, a residual, and a no-op + (-0.0) left by an earlier rewrite.
// The matmul dominates, so this measures what the passes save when the
// elementwise work is NOT most of the runtime.
func.func @kernel(%x: tensor<256x256xf32>, %w: tensor<256x256xf32>,
                  %bias: tensor<256x256xf32>)
    -> tensor<256x256xf32> attributes {llvm.emit_c_interface} {
  %half = topt.constant dense<0.5> : tensor<256x256xf32>
  %two = topt.constant dense<2.0> : tensor<256x256xf32>
  %nz = topt.constant dense<-0.0> : tensor<256x256xf32>
  %h = topt.matmul %x, %w : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
  %ht = topt.transpose %h, [1, 0] : tensor<256x256xf32> -> tensor<256x256xf32>
  %hh = topt.transpose %ht, [1, 0] : tensor<256x256xf32> -> tensor<256x256xf32>
  %scale = topt.mul %half, %two : tensor<256x256xf32>
  %s = topt.mul %hh, %scale : tensor<256x256xf32>
  %b = topt.add %s, %bias : tensor<256x256xf32>
  %r = topt.add %b, %x : tensor<256x256xf32>
  %a = topt.mul %r, %r : tensor<256x256xf32>
  %o = topt.add %a, %nz : tensor<256x256xf32>
  return %o : tensor<256x256xf32>
}
