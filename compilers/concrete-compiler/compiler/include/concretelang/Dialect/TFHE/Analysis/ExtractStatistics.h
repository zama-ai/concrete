// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_TFHE_ANALYSIS_EXTRACT_STATISTICS_H
#define CONCRETELANG_DIALECT_TFHE_ANALYSIS_EXTRACT_STATISTICS_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <concretelang/Support/CompilationFeedback.h>

namespace mlir {
namespace concretelang {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createStatisticExtractionPass(ProgramCompilationFeedback &feedback);
} // namespace concretelang
} // namespace mlir

#endif
