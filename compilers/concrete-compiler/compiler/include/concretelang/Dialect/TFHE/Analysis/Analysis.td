#ifndef CONCRETELANG_DIALECT_TFHE_ANALYSIS
#define CONCRETELANG_DIALECT_TFHE_ANALYSIS

include "mlir/Pass/PassBase.td"

def ExtractStatistics : Pass<"ExtractStatistics", "::mlir::ModuleOp"> {
  let summary = "Extracts statistics";
  let description = [{
    Extracts different statistics (e.g. number of certain crypto operations),
    and provides those numbers throught the CompilationFeedback.
  }];
}

#endif
