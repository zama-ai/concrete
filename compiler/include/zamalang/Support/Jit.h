#ifndef COMPILER_JIT_H
#define COMPILER_JIT_H

#include "zamalang/Support/CompilerTools.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir {
namespace zamalang {
mlir::LogicalResult
runJit(mlir::ModuleOp module, llvm::StringRef func,
       llvm::ArrayRef<uint64_t> funcArgs, mlir::zamalang::KeySet &keySet,
       std::function<llvm::Error(llvm::Module *)> optPipeline,
       llvm::raw_ostream &os);
} // namespace zamalang
} // namespace mlir

#endif // COMPILER_JIT_H
