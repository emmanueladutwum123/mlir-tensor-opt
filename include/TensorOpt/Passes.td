#ifndef TENSOROPT_PASSES_TD
#define TENSOROPT_PASSES_TD

include "mlir/Pass/PassBase.td"

def ToptConstantFold : Pass<"topt-constant-fold", "func::FuncOp"> {
  let summary = "Evaluate topt ops whose operands are all constants";
  let description = [{
    Runs the ops' fold() hooks to a fixpoint and nothing else: no rewrite
    patterns, so no algebraic simplification. Folded results are rebuilt as
    topt.constant by the dialect's constant materializer, and constants left
    without uses are erased.
  }];
}

def ToptSimplify : Pass<"topt-simplify", "func::FuncOp"> {
  let summary = "Apply algebraic identities that hold exactly in IEEE-754";
  let description = [{
    transpose(transpose(x, p), q) -> transpose(x, p o q), and -> x when the
    composition is the identity; transpose(x, identity) -> x;
    x * 1.0 -> x; x + (-0.0) -> x.

    Deliberately absent, because they are false for floats: x + 0.0 -> x
    (wrong for x = -0.0) and x * 0.0 -> 0.0 (wrong for NaN, inf and
    negative x). Folding is disabled in this pass so that it simplifies and
    does not also evaluate constants.
  }];
}

def ToptFuseElementwise : Pass<"topt-fuse-elementwise", "func::FuncOp"> {
  let summary = "Fuse chains and trees of elementwise ops into one op";
  let description = [{
    Merges an elementwise producer (add, mul, fused_elementwise) into its
    elementwise consumer when the consumer is the producer's only user. The
    result is a topt.fused_elementwise whose body evaluates the producer's
    scalar code and then the consumer's. Repeated to a fixpoint, an n-op
    chain becomes one op, which lowers to one loop nest and no intermediate
    tensors.

    Rejected: fusing producers that have other users. That either
    recomputes the producer inside every consumer (more flops) or needs a
    multi-result fused op (more complexity); this pass does neither.
  }];
  let dependentDialects = ["arith::ArithDialect"];
}

#endif // TENSOROPT_PASSES_TD
