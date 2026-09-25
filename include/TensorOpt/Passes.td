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

#endif // TENSOROPT_PASSES_TD
