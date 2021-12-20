#ifndef ZAMALANG_SUPPORT_PIPELINE_H_
#define ZAMALANG_SUPPORT_PIPELINE_H_

#include <llvm/IR/Module.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <zamalang/Support/V0Parameters.h>

namespace mlir {
namespace zamalang {
namespace pipeline {

mlir::LogicalResult autopar(mlir::MLIRContext &context, mlir::ModuleOp &module,
                            std::function<bool(mlir::Pass *)> enablePass);

llvm::Expected<llvm::Optional<mlir::zamalang::V0FHEConstraint>>
getFHEConstraintsFromHLFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                           std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
tileMarkedHLFHELinalg(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
markHLFHELinalgForTiling(mlir::MLIRContext &context, mlir::ModuleOp &module,
                         llvm::ArrayRef<int64_t> tileSizes,
                         std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
lowerHLFHEToMidLFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
lowerMidLFHEToLowLFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      llvm::Optional<V0FHEContext> &fheContext,
                      std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
lowerLowLFHEToStd(mlir::MLIRContext &context, mlir::ModuleOp &module,
                  std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult
lowerStdToLLVMDialect(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass);

mlir::LogicalResult optimizeLLVMModule(llvm::LLVMContext &llvmContext,
                                       llvm::Module &module);

std::unique_ptr<llvm::Module>
lowerLLVMDialectToLLVMIR(mlir::MLIRContext &context,
                         llvm::LLVMContext &llvmContext,
                         mlir::ModuleOp &module);

} // namespace pipeline
} // namespace zamalang
} // namespace mlir

#endif
