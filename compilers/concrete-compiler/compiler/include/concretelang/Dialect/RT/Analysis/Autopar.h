// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_RT_ANALYSIS_AUTOPAR_H
#define CONCRETELANG_DIALECT_RT_ANALYSIS_AUTOPAR_H

#include <concretelang/Dialect/RT/IR/RTOps.h>
#include <functional>
#include <mlir/Pass/Pass.h>

namespace mlir {

class LLVMTypeConverter;
class BufferizeTypeConverter;
class RewritePatternSet;

namespace concretelang {
std::unique_ptr<mlir::Pass> createBuildDataflowTaskGraphPass();
std::unique_ptr<mlir::Pass> createLowerDataflowTasksPass();
std::unique_ptr<mlir::Pass> createBufferizeDataflowTaskOpsPass();
std::unique_ptr<mlir::Pass> createFinalizeTaskCreationPass();
std::unique_ptr<mlir::Pass> createStartStopPass(bool useOMP);
std::unique_ptr<mlir::Pass> createFixupBufferDeallocationPass();
void populateRTToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                        mlir::RewritePatternSet &patterns);
void populateRTBufferizePatterns(mlir::BufferizeTypeConverter &typeConverter,
                                 mlir::RewritePatternSet &patterns);
} // namespace concretelang
} // namespace mlir

#endif
