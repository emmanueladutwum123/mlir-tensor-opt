#ifndef TENSOROPT_DIALECT_TD
#define TENSOROPT_DIALECT_TD

include "mlir/IR/OpBase.td"

def TensorOpt_Dialect : Dialect {
  let name = "topt";
  let cppNamespace = "::mlir::topt";
  let summary = "A small tensor dialect for studying optimisation passes";
  let description = [{
    Value-semantic ops on statically shaped f32 tensors. The dialect says what
    is computed, never how: no iteration spaces, no memory. It sits above
    `linalg`, and `convert-topt-to-linalg` lowers it there.
  }];
  // Folding returns attributes; the dialect turns them back into ops.
  let hasConstantMaterializer = 1;
}

class TensorOpt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorOpt_Dialect, mnemonic, traits>;

// Every tensor in this dialect is a ranked, statically shaped f32 tensor.
// Rejected: dynamic shapes. They would force shape arithmetic
// (tensor.dim, dynamic tensor.empty) into every lowering and verifier for no
// gain in what the passes demonstrate.
def TensorOpt_Tensor : StaticShapeTensorOf<[F32]>;

// Dense f32 payload of topt.constant.
def TensorOpt_F32ElementsAttr : FloatElementsAttr<32>;

#endif // TENSOROPT_DIALECT_TD
