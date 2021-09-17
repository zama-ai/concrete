#ifndef ZAMALANG_SUPPORT_PIPELINE_H_
#define ZAMALANG_SUPPORT_PIPELINE_H_

#include <llvm/IR/Module.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Support/LogicalResult.h>
#include <zamalang/Support/V0Parameters.h>

namespace mlir {
namespace zamalang {
namespace pipeline {

mlir::LogicalResult lowerHLFHEToMidLFHE(mlir::MLIRContext &context,
                                        mlir::ModuleOp &module, bool verbose);

mlir::LogicalResult lowerMidLFHEToLowLFHE(mlir::MLIRContext &context,
                                          mlir::ModuleOp &module,
                                          V0FHEContext &fheContext,
                                          bool parametrize);

mlir::LogicalResult lowerLowLFHEToStd(mlir::MLIRContext &context,
                                      mlir::ModuleOp &module);

mlir::LogicalResult lowerStdToLLVMDialect(mlir::MLIRContext &context,
                                          mlir::ModuleOp &module, bool verbose);

mlir::LogicalResult optimizeLLVMModule(llvm::LLVMContext &llvmContext,
                                       llvm::Module &module);

mlir::LogicalResult lowerHLFHEToStd(mlir::MLIRContext &context,
                                    mlir::ModuleOp &module,
                                    V0FHEContext &fheContext, bool verbose);

std::unique_ptr<llvm::Module>
lowerLLVMDialectToLLVMIR(mlir::MLIRContext &context,
                         llvm::LLVMContext &llvmContext,
                         mlir::ModuleOp &module);
} // namespace pipeline
} // namespace zamalang
} // namespace mlir

#endif
