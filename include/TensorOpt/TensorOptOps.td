#ifndef TENSOROPT_OPS_TD
#define TENSOROPT_OPS_TD

include "TensorOpt/TensorOptDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// constant
//===----------------------------------------------------------------------===//

def TensorOpt_ConstantOp : TensorOpt_Op<"constant",
    [ConstantLike, Pure, AllTypesMatch<["value", "output"]>]> {
  let summary = "A tensor whose value is known at compile time";
  let arguments = (ins TensorOpt_F32ElementsAttr:$value);
  let results = (outs TensorOpt_Tensor:$output);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Elementwise binary ops
//===----------------------------------------------------------------------===//

// Rejected: implicit broadcasting (numpy-style). It makes every elementwise
// op carry an implicit reshape, so fusion would need indexing maps just to
// say which element of each operand is read. Same shapes everywhere keeps
// fusion a pure composition of scalar functions.
class TensorOpt_BinaryElementwiseOp<string mnemonic, list<Trait> traits = []>
    : TensorOpt_Op<mnemonic,
        !listconcat(traits, [Pure, SameOperandsAndResultType])> {
  let arguments = (ins TensorOpt_Tensor:$lhs, TensorOpt_Tensor:$rhs);
  let results = (outs TensorOpt_Tensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
  let hasFolder = 1;
}

def TensorOpt_AddOp : TensorOpt_BinaryElementwiseOp<"add", [Commutative]> {
  let summary = "Elementwise IEEE-754 single-precision addition";
}

def TensorOpt_MulOp : TensorOpt_BinaryElementwiseOp<"mul", [Commutative]> {
  let summary = "Elementwise IEEE-754 single-precision multiplication";
}

//===----------------------------------------------------------------------===//
// matmul
//===----------------------------------------------------------------------===//

def TensorOpt_MatmulOp : TensorOpt_Op<"matmul", [Pure]> {
  let summary = "Rank-2 matrix product: (MxK) x (KxN) -> (MxN)";
  let arguments = (ins TensorOpt_Tensor:$lhs, TensorOpt_Tensor:$rhs);
  let results = (outs TensorOpt_Tensor:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// transpose
//===----------------------------------------------------------------------===//

def TensorOpt_TransposeOp : TensorOpt_Op<"transpose", [Pure]> {
  let summary = "Permute the dimensions of a tensor";
  let description = [{
    Same convention as `linalg.transpose`:
    `dim(output, i) == dim(input, permutation[i])`.

    Rejected: a rank-2-only transpose with no attribute. That would make
    `transpose(transpose(x)) -> x` trivially true and teach nothing. With an
    explicit permutation the simplification is permutation *composition*,
    and it only deletes the pair when the composition is the identity.
  }];
  let arguments = (ins TensorOpt_Tensor:$input,
                       DenseI64ArrayAttr:$permutation);
  let results = (outs TensorOpt_Tensor:$output);
  let assemblyFormat = [{
    $input `,` $permutation attr-dict `:` type($input) `->` type($output)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// fused_elementwise: produced only by the fusion pass
//===----------------------------------------------------------------------===//

def TensorOpt_FusedElementwiseOp : TensorOpt_Op<"fused_elementwise",
    [Pure, IsolatedFromAbove, SingleBlock]> {
  let summary = "One pointwise map over N same-shaped tensors";
  let description = [{
    The body receives one f32 scalar per input and yields one f32. The op
    applies the body at every index of the (shared) iteration space. It is
    what the fusion pass builds from a chain of add/mul ops, and it lowers to
    exactly one `linalg.generic`, which is one loop nest.

    Rejected: fusing straight into `linalg.generic` inside the fusion pass.
    That would couple an optimisation to a lowering decision, and the fused
    form could no longer be optimised or even pretty-printed at the `topt`
    level.
  }];
  let arguments = (ins Variadic<TensorOpt_Tensor>:$inputs);
  let results = (outs TensorOpt_Tensor:$output);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs) `->` type($output) $body
  }];
  let hasVerifier = 1;
}

def TensorOpt_YieldOp : TensorOpt_Op<"yield",
    [Pure, Terminator, HasParent<"FusedElementwiseOp">]> {
  let summary = "Yield the scalar result of a fused_elementwise body";
  let arguments = (ins F32:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // TENSOROPT_OPS_TD
