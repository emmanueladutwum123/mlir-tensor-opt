#include "TensorOpt/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::topt {
#define GEN_PASS_DEF_TOPTCONSTANTFOLD
#include "TensorOpt/Passes.h.inc"

namespace {
struct ConstantFoldPass
    : impl::ToptConstantFoldBase<ConstantFoldPass> {
  void runOnOperation() override {
    // An empty pattern set: the greedy driver still calls every op's fold()
    // and materializes the attributes it returns, which is all this pass is.
    // Rejected: a hand-written worklist. It would re-implement exactly what
    // the driver does (revisit users of a folded op, erase dead constants,
    // deduplicate equal constants) with more places to get it wrong.
    RewritePatternSet patterns(&getContext());
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::topt
