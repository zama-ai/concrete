// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_DIALECT_RT_ANALYSIS_AUTOPAR_H
#define ZAMALANG_DIALECT_RT_ANALYSIS_AUTOPAR_H

#include <functional>
#include <mlir/Pass/Pass.h>
#include <zamalang/Dialect/RT/IR/RTOps.h>

namespace mlir {

class LLVMTypeConverter;
class BufferizeTypeConverter;
class RewritePatternSet;

namespace zamalang {
std::unique_ptr<mlir::Pass>
createBuildDataflowTaskGraphPass(bool debug = false);
std::unique_ptr<mlir::Pass> createLowerDataflowTasksPass(bool debug = false);
std::unique_ptr<mlir::Pass>
createBufferizeDataflowTaskOpsPass(bool debug = false);
std::unique_ptr<mlir::Pass> createFixupDataflowTaskOpsPass(bool debug = false);
void populateRTToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                        mlir::RewritePatternSet &patterns);
void populateRTBufferizePatterns(mlir::BufferizeTypeConverter &typeConverter,
                                 mlir::RewritePatternSet &patterns);
} // namespace zamalang
} // namespace mlir

#endif
