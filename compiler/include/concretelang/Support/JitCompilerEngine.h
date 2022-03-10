// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_JIT_COMPILER_ENGINE_H
#define CONCRETELANG_SUPPORT_JIT_COMPILER_ENGINE_H

#include "concretelang/ClientLib/KeySetCache.h"
#include <concretelang/Support/CompilerEngine.h>
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Jit.h>
#include <concretelang/Support/LambdaArgument.h>
#include <concretelang/Support/LambdaSupport.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir {
namespace concretelang {

using ::concretelang::clientlib::KeySetCache;
namespace clientlib = ::concretelang::clientlib;

// A compiler engine that JIT-compiles a source and produces a lambda
// object directly invocable through its call operator.
class JitCompilerEngine : public CompilerEngine {
public:
  // Wrapper class around `JITLambda` and `JITLambda::Argument` that
  // allows for direct invocation of a compiled function through
  // `operator ()`.
  class Lambda {
  public:
    Lambda(Lambda &&other)
        : innerLambda(std::move(other.innerLambda)),
          keySet(std::move(other.keySet)),
          compilationContext(other.compilationContext),
          clientParameters(other.clientParameters) {}

    Lambda(std::shared_ptr<CompilationContext> compilationContext,
           std::unique_ptr<JITLambda> lambda, std::unique_ptr<KeySet> keySet,
           clientlib::ClientParameters clientParameters)
        : innerLambda(std::move(lambda)), keySet(std::move(keySet)),
          compilationContext(compilationContext),
          clientParameters(clientParameters) {}

    // Returns the number of arguments required for an invocation of
    // the lambda
    size_t getNumArguments() { return this->keySet->numInputs(); }

    // Returns the number of results an invocation of the lambda
    // produces
    size_t getNumResults() { return this->keySet->numOutputs(); }

    // Invocation with an dynamic list of arguments of different
    // types, specified as `LambdaArgument`s
    template <typename ResT = uint64_t>
    llvm::Expected<ResT> operator()(llvm::ArrayRef<LambdaArgument *> args) {
      auto publicArguments = LambdaArgumentAdaptor::exportArguments(
          args, clientParameters, *this->keySet);

      if (auto err = publicArguments.takeError()) {
        return err;
      }

      // Call the lambda
      auto publicResult = this->innerLambda->call(**publicArguments);
      if (auto err = publicResult.takeError()) {
        return std::move(err);
      }

      return typedResult<ResT>(*keySet, **publicResult);
    }

    // Invocation with an array of arguments of the same type
    template <typename T, typename ResT = uint64_t>
    llvm::Expected<ResT> operator()(const llvm::ArrayRef<T> args) {
      // Encrypt the arguments
      auto encryptedArgs = clientlib::EncryptedArguments::create(*keySet, args);
      if (encryptedArgs.has_error()) {
        return StreamStringError(encryptedArgs.error().mesg);
      }

      // Export as public arguments
      auto publicArguments = encryptedArgs.value()->exportPublicArguments(
          clientParameters, keySet->runtimeContext());
      if (!publicArguments.has_value()) {
        return StreamStringError(publicArguments.error().mesg);
      }

      // Call the lambda
      auto publicResult = this->innerLambda->call(*publicArguments.value());
      if (auto err = publicResult.takeError()) {
        return std::move(err);
      }

      return typedResult<ResT>(*keySet, **publicResult);
    }

    // Invocation with arguments of different types
    template <typename ResT = uint64_t, typename... Ts>
    llvm::Expected<ResT> operator()(const Ts... ts) {
      // Encrypt the arguments
      auto encryptedArgs =
          clientlib::EncryptedArguments::create(*keySet, ts...);

      if (encryptedArgs.has_error()) {
        return StreamStringError(encryptedArgs.error().mesg);
      }

      // Export as public arguments
      auto publicArguments = encryptedArgs.value()->exportPublicArguments(
          clientParameters, keySet->runtimeContext());
      if (!publicArguments.has_value()) {
        return StreamStringError(publicArguments.error().mesg);
      }

      // Call the lambda
      auto publicResult = this->innerLambda->call(*publicArguments.value());
      if (auto err = publicResult.takeError()) {
        return std::move(err);
      }

      return typedResult<ResT>(*keySet, **publicResult);
    }

  protected:
    std::unique_ptr<JITLambda> innerLambda;
    std::unique_ptr<KeySet> keySet;
    std::shared_ptr<CompilationContext> compilationContext;
    const clientlib::ClientParameters clientParameters;
  };

  JitCompilerEngine(std::shared_ptr<CompilationContext> compilationContext =
                        CompilationContext::createShared(),
                    unsigned int optimizationLevel = 3);

  /// Build a Lambda from a source MLIR, with `funcName` as entrypoint.
  /// Use runtimeLibPath as a shared library if specified.
  llvm::Expected<Lambda>
  buildLambda(llvm::StringRef src, llvm::StringRef funcName = "main",
              llvm::Optional<KeySetCache> cachePath = {},
              llvm::Optional<llvm::StringRef> runtimeLibPath = {});

  llvm::Expected<Lambda>
  buildLambda(std::unique_ptr<llvm::MemoryBuffer> buffer,
              llvm::StringRef funcName = "main",
              llvm::Optional<KeySetCache> cachePath = {},
              llvm::Optional<llvm::StringRef> runtimeLibPath = {});

  llvm::Expected<Lambda>
  buildLambda(llvm::SourceMgr &sm, llvm::StringRef funcName = "main",
              llvm::Optional<KeySetCache> cachePath = {},
              llvm::Optional<llvm::StringRef> runtimeLibPath = {});

protected:
  llvm::Expected<mlir::LLVM::LLVMFuncOp> findLLVMFuncOp(mlir::ModuleOp module,
                                                        llvm::StringRef name);
  unsigned int optimizationLevel;
};

} // namespace concretelang
} // namespace mlir

#endif
