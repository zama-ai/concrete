#include "llvm/Support/Error.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <zamalang/Support/JitCompilerEngine.h>

namespace mlir {
namespace zamalang {

JitCompilerEngine::JitCompilerEngine(
    std::shared_ptr<CompilationContext> compilationContext,
    unsigned int optimizationLevel)
    : CompilerEngine(compilationContext), optimizationLevel(optimizationLevel) {
}

// Returns the `LLVMFuncOp` operation in the compiled module with the
// specified name. If no LLVMFuncOp with that name exists or if there
// was no prior call to `compile()` resulting in an MLIR module in the
// LLVM dialect, an error is returned.
llvm::Expected<mlir::LLVM::LLVMFuncOp>
JitCompilerEngine::findLLVMFuncOp(mlir::ModuleOp module, llvm::StringRef name) {
  auto funcOps = module.getOps<mlir::LLVM::LLVMFuncOp>();
  auto funcOp = llvm::find_if(
      funcOps, [&](mlir::LLVM::LLVMFuncOp op) { return op.getName() == name; });

  if (funcOp == funcOps.end()) {
    return StreamStringError()
           << "Module does not contain function named '" << name.str() << "'";
  }

  return *funcOp;
}

// Build a lambda from the function with the name given in
// `funcName` from the sources in `buffer`.
llvm::Expected<JitCompilerEngine::Lambda>
JitCompilerEngine::buildLambda(std::unique_ptr<llvm::MemoryBuffer> buffer,
                               llvm::StringRef funcName,
                               llvm::Optional<llvm::StringRef> runtimeLibPath) {
  llvm::SourceMgr sm;

  sm.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  llvm::Expected<JitCompilerEngine::Lambda> res =
      this->buildLambda(sm, funcName, runtimeLibPath);

  return std::move(res);
}

// Build a lambda from the function with the name given in `funcName`
// from the source string `s`.
llvm::Expected<JitCompilerEngine::Lambda>
JitCompilerEngine::buildLambda(llvm::StringRef s, llvm::StringRef funcName,
                               llvm::Optional<llvm::StringRef> runtimeLibPath) {
  std::unique_ptr<llvm::MemoryBuffer> mb = llvm::MemoryBuffer::getMemBuffer(s);
  llvm::Expected<JitCompilerEngine::Lambda> res =
      this->buildLambda(std::move(mb), funcName, runtimeLibPath);

  return std::move(res);
}

// Build a lambda from the function with the name given in
// `funcName` from the sources managed by the source manager `sm`.
llvm::Expected<JitCompilerEngine::Lambda>
JitCompilerEngine::buildLambda(llvm::SourceMgr &sm, llvm::StringRef funcName,
                               llvm::Optional<llvm::StringRef> runtimeLibPath) {
  MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();

  this->setGenerateClientParameters(true);
  this->setClientParametersFuncName(funcName);

  // First, compile to LLVM Dialect
  llvm::Expected<CompilerEngine::CompilationResult> compResOrErr =
      this->compile(sm, Target::LLVM_IR);

  if (!compResOrErr)
    return std::move(compResOrErr.takeError());

  mlir::ModuleOp module = compResOrErr->mlirModuleRef->get();

  // Locate function to JIT-compile
  llvm::Expected<mlir::LLVM::LLVMFuncOp> funcOrError =
      this->findLLVMFuncOp(compResOrErr->mlirModuleRef->get(), funcName);

  if (!funcOrError)
    return std::move(funcOrError.takeError());

  // Prepare LLVM infrastructure for JIT compilation
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerLLVMDialectTranslation(mlirContext);

  llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline =
      mlir::makeOptimizingTransformer(3, 0, nullptr);

  llvm::Expected<std::unique_ptr<JITLambda>> lambdaOrErr =
      mlir::zamalang::JITLambda::create(funcName, module, optPipeline,
                                        runtimeLibPath);

  // Generate the KeySet for encrypting lambda arguments, decrypting lambda
  // results
  if (!compResOrErr->clientParameters.hasValue()) {
    return StreamStringError("Cannot generate the keySet since client "
                             "parameters has not been computed");
  }

  llvm::Expected<std::unique_ptr<mlir::zamalang::KeySet>> keySetOrErr =
      mlir::zamalang::KeySet::generate(*compResOrErr->clientParameters, 0, 0);

  if (auto err = keySetOrErr.takeError())
    return std::move(err);

  if (!lambdaOrErr)
    return std::move(lambdaOrErr.takeError());

  return Lambda{this->compilationContext, std::move(lambdaOrErr.get()),
                std::move(*keySetOrErr)};
}

} // namespace zamalang
} // namespace mlir
