// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_PIPELINE_H_
#define CONCRETELANG_SUPPORT_PIPELINE_H_

#include "concretelang/Support/V0Parameters.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Module.h"

namespace mlir {
namespace concretelang {
namespace pipeline {

mlir::LogicalResult autopar(mlir::MLIRContext &context, mlir::ModuleOp &module,
                            std::function<bool(mlir::Pass *)> enablePass);

llvm::Expected<std::map<std::string, std::optional<optimizer::Description>>>
getFHEContextFromFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                     optimizer::Config config,
                     std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
markFHELinalgForTiling(mlir::MLIRContext &context, mlir::ModuleOp &module,
                       llvm::ArrayRef<int64_t> tileSizes,
                       std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
transformHighLevelFHEOps(mlir::MLIRContext &context, mlir::ModuleOp &module,
                         std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
lowerFHELinalgToLinalg(mlir::MLIRContext &context, mlir::ModuleOp &module,
                       std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
tileMarkedLinalg(mlir::MLIRContext &context, mlir::ModuleOp &module,
                 std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
lowerLinalgToLoops(mlir::MLIRContext &context, mlir::ModuleOp &module,
                   std::function<bool(mlir::Pass *)> enablePass,
                   bool parallelizeLoops);

mlir::LogicalResult
transformFHEBoolean(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
transformFHEBigInt(mlir::MLIRContext &context, mlir::ModuleOp &module,
                   std::function<bool(mlir::Pass *)> enablePass,
                   unsigned int chunkSize, unsigned int chunkWidth);

mlir::LogicalResult
lowerFHEToTFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
               std::optional<V0FHEContext> &fheContext,
               std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
parametrizeTFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                std::optional<V0FHEContext> &fheContext,
                std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult batchTFHE(mlir::MLIRContext &context,
                              mlir::ModuleOp &module,
                              std::function<bool(mlir::Pass *)> enablePass,
                              int64_t maxBatchSize);

mlir::LogicalResult
normalizeTFHEKeys(mlir::MLIRContext &context, mlir::ModuleOp &module,
                  std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
extractTFHEStatistics(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass,
                      CompilationFeedback &feedback);

mlir::LogicalResult
lowerTFHEToConcrete(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
computeMemoryUsage(mlir::MLIRContext &context, mlir::ModuleOp &module,
                   std::function<bool(mlir::Pass *)> enablePass,
                   CompilationFeedback &feedback);

mlir::LogicalResult
lowerConcreteLinalgToLoops(mlir::MLIRContext &context, mlir::ModuleOp &module,
                           std::function<bool(mlir::Pass *)> enablePass,
                           bool parallelizeLoops);

mlir::LogicalResult optimizeTFHE(mlir::MLIRContext &context,
                                 mlir::ModuleOp &module,
                                 std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult simulateTFHE(mlir::MLIRContext &context,
                                 mlir::ModuleOp &module,
                                 std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult extractSDFGOps(mlir::MLIRContext &context,
                                   mlir::ModuleOp &module,
                                   std::function<bool(mlir::Pass *)> enablePass,
                                   bool unrollLoops);

mlir::LogicalResult
addRuntimeContext(mlir::MLIRContext &context, mlir::ModuleOp &module,
                  std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
lowerSDFGToStd(mlir::MLIRContext &context, mlir::ModuleOp &module,
               std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
lowerStdToLLVMDialect(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult lowerToStd(mlir::MLIRContext &context,
                               mlir::ModuleOp &module,
                               std::function<bool(mlir::Pass *)> enablePass,
                               bool parallelizeLoops);

mlir::LogicalResult lowerToCAPI(mlir::MLIRContext &context,
                                mlir::ModuleOp &module,
                                std::function<bool(mlir::Pass *)> enablePass,
                                bool gpu);

mlir::LogicalResult optimizeLLVMModule(llvm::LLVMContext &llvmContext,
                                       llvm::Module &module);

std::unique_ptr<llvm::Module>
lowerLLVMDialectToLLVMIR(mlir::MLIRContext &context,
                         llvm::LLVMContext &llvmContext,
                         mlir::ModuleOp &module);

} // namespace pipeline
} // namespace concretelang
} // namespace mlir

#endif
