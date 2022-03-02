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
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir {
namespace concretelang {

using ::concretelang::clientlib::KeySetCache;
namespace clientlib = ::concretelang::clientlib;

namespace {
// Generic function template as well as specializations of
// `typedResult` must be declared at namespace scope due to return
// type template specialization

// Helper function for `JitCompilerEngine::Lambda::operator()`
// implementing type-dependent preparation of the result.
template <typename ResT>
llvm::Expected<ResT> typedResult(clientlib::KeySet &keySet,
                                 clientlib::PublicResult &result);

// Specialization of `typedResult()` for scalar results, forwarding
// scalar value to caller
template <>
inline llvm::Expected<uint64_t> typedResult(clientlib::KeySet &keySet,
                                            clientlib::PublicResult &result) {
  auto clearResult = result.asClearTextVector(keySet, 0);
  if (!clearResult.has_value()) {
    return StreamStringError("typedResult cannot get clear text vector")
           << clearResult.error().mesg;
  }
  if (clearResult.value().size() != 1) {
    return StreamStringError("typedResult expect only one value but got ")
           << clearResult.value().size();
  }
  return clearResult.value()[0];
}

template <typename T>
inline llvm::Expected<std::vector<T>>
typedVectorResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  auto clearResult = result.asClearTextVector(keySet, 0);
  if (!clearResult.has_value()) {
    return StreamStringError("typedVectorResult cannot get clear text vector")
           << clearResult.error().mesg;
  }
  return std::move(clearResult.value());
}

// Specializations of `typedResult()` for vector results, initializing
// an `std::vector` of the right size with the results and forwarding
// it to the caller with move semantics.
//
// Cannot factor out into a template template <typename T> inline
// llvm::Expected<std::vector<uint8_t>>
// typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result); due
// to ambiguity with scalar template
// template <>
// inline llvm::Expected<std::vector<uint8_t>>
// typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
//   return typedVectorResult<uint8_t>(keySet, result);
// }
// template <>
// inline llvm::Expected<std::vector<uint16_t>>
// typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
//   return typedVectorResult<uint16_t>(keySet, result);
// }
// template <>
// inline llvm::Expected<std::vector<uint32_t>>
// typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
//   return typedVectorResult<uint32_t>(keySet, result);
// }
template <>
inline llvm::Expected<std::vector<uint64_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<uint64_t>(keySet, result);
}

template <typename T>
llvm::Expected<std::unique_ptr<LambdaArgument>>
buildTensorLambdaResult(clientlib::KeySet &keySet,
                        clientlib::PublicResult &result) {
  llvm::Expected<std::vector<T>> tensorOrError =
      typedResult<std::vector<T>>(keySet, result);

  if (auto err = tensorOrError.takeError())
    return std::move(err);
  std::vector<int64_t> tensorDim(result.buffers[0].sizes.begin(),
                                 result.buffers[0].sizes.end() - 1);

  return std::make_unique<TensorLambdaArgument<IntLambdaArgument<T>>>(
      *tensorOrError, tensorDim);
}

// Specialization of `typedResult()` for a single result wrapped into
// a `LambdaArgument`.
template <>
inline llvm::Expected<std::unique_ptr<LambdaArgument>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  auto gate = keySet.outputGate(0);
  // scalar case
  if (gate.shape.dimensions.empty()) {
    auto clearResult = result.asClearTextVector(keySet, 0);
    if (clearResult.has_error()) {
      return StreamStringError("typedResult: ") << clearResult.error().mesg;
    }
    auto res = clearResult.value()[0];

    return std::make_unique<IntLambdaArgument<uint64_t>>(res);
  }
  // tensor case
  // auto width = gate.shape.width;

  // if (width > 32)
  return buildTensorLambdaResult<uint64_t>(keySet, result);
  // else if (width > 16)
  //   return buildTensorLambdaResult<uint32_t>(keySet, result);
  // else if (width > 8)
  //   return buildTensorLambdaResult<uint16_t>(keySet, result);
  // else if (width <= 8)
  //   return buildTensorLambdaResult<uint8_t>(keySet, result);

  // return StreamStringError("Cannot handle scalars with more than 64 bits");
}

} // namespace

// Adaptor class that push arguments specified as instances of
// `LambdaArgument` to `clientlib::EncryptedArguments`.
class JITLambdaArgumentAdaptor {
public:
  // Checks if the argument `arg` is an plaintext / encrypted integer
  // argument or a plaintext / encrypted tensor argument with a
  // backing integer type `IntT` and push the argument to `encryptedArgs`.
  //
  // Returns `true` if `arg` has one of the types above and its value
  // was successfully added to `encryptedArgs`, `false` if none of the types
  // matches or an error if a type matched, but adding the argument to
  // `encryptedArgs` failed.
  template <typename IntT>
  static inline llvm::Expected<bool>
  tryAddArg(clientlib::EncryptedArguments &encryptedArgs,
            const LambdaArgument &arg, clientlib::KeySet &keySet) {
    if (auto ila = arg.dyn_cast<IntLambdaArgument<IntT>>()) {
      auto res = encryptedArgs.pushArg(ila->getValue(), keySet);
      if (!res.has_value()) {
        return StreamStringError(res.error().mesg);
      } else {
        return true;
      }
    } else if (auto tla = arg.dyn_cast<
                          TensorLambdaArgument<IntLambdaArgument<IntT>>>()) {
      auto res =
          encryptedArgs.pushArg(tla->getValue(), tla->getDimensions(), keySet);
      if (!res.has_value()) {
        return StreamStringError(res.error().mesg);
      } else {
        return true;
      }
    }
    return false;
  }

  // Recursive case for `tryAddArg<IntT>(...)`
  template <typename IntT, typename NextIntT, typename... IntTs>
  static inline llvm::Expected<bool>
  tryAddArg(clientlib::EncryptedArguments &encryptedArgs,
            const LambdaArgument &arg, clientlib::KeySet &keySet) {
    llvm::Expected<bool> successOrError =
        tryAddArg<IntT>(encryptedArgs, arg, keySet);

    if (!successOrError)
      return successOrError.takeError();

    if (successOrError.get() == false)
      return tryAddArg<NextIntT, IntTs...>(encryptedArgs, arg, keySet);
    else
      return true;
  }

  // Attempts to push a single argument `arg` to `encryptedArgs`. Returns an
  // error if either the argument type is unsupported or if the argument types
  // is supported, but adding it to `encryptedArgs` failed.
  static inline llvm::Error
  addArgument(clientlib::EncryptedArguments &encryptedArgs,
              const LambdaArgument &arg, clientlib::KeySet &keySet) {
    // Try the supported integer types; size_t needs explicit
    // treatment, since it may alias none of the fixed size integer
    // types
    llvm::Expected<bool> successOrError =
        JITLambdaArgumentAdaptor::tryAddArg<uint64_t, uint32_t, uint16_t,
                                            uint8_t, size_t>(encryptedArgs, arg,
                                                             keySet);

    if (!successOrError)
      return successOrError.takeError();

    if (successOrError.get() == false)
      return StreamStringError("Unknown argument type");
    else
      return llvm::Error::success();
  }
};

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
    llvm::Expected<ResT>
    operator()(llvm::ArrayRef<LambdaArgument *> lambdaArgs) {
      // Encrypt the arguments
      auto encryptedArgs = clientlib::EncryptedArguments::empty();

      for (size_t i = 0; i < lambdaArgs.size(); i++) {
        if (llvm::Error err = JITLambdaArgumentAdaptor::addArgument(
                *encryptedArgs, *lambdaArgs[i], *this->keySet)) {
          return std::move(err);
        }
      }

      auto check = encryptedArgs->checkAllArgs(*this->keySet);
      if (check.has_error()) {
        return StreamStringError(check.error().mesg);
      }

      // Export as public arguments
      auto publicArguments = encryptedArgs->exportPublicArguments(
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

    // Invocation with an array of arguments of the same type
    template <typename T, typename ResT = uint64_t>
    llvm::Expected<ResT> operator()(const llvm::ArrayRef<T> args) {
      // Encrypt the arguments
      auto encryptedArgs = clientlib::EncryptedArguments::empty();

      for (size_t i = 0; i < args.size(); i++) {
        auto res = encryptedArgs->pushArg(args[i], *keySet);
        if (res.has_error()) {
          return StreamStringError(res.error().mesg);
        }
      }

      auto check = encryptedArgs->checkAllArgs(*this->keySet);
      if (check.has_error()) {
        return StreamStringError(check.error().mesg);
      }

      // Export as public arguments
      auto publicArguments = encryptedArgs->exportPublicArguments(
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
    template <int pos>
    inline llvm::Error addArgs(JITLambda::Argument *jitArgs) {
      // base case -- nothing to do
      return llvm::Error::success();
    }

    // Recursive case for scalars: extract first scalar argument from
    // parameter pack and forward rest
    template <int pos, typename ArgT, typename... Ts>
    inline llvm::Error addArgs(JITLambda::Argument *jitArgs, ArgT arg,
                               Ts... remainder) {
      if (auto err = jitArgs->setArg(pos, arg)) {
        return StreamStringError()
               << "Cannot push scalar argument " << pos << ": " << err;
      }

      return this->addArgs<pos + 1>(jitArgs, remainder...);
    }

    // Recursive case for tensors: extract pointer and size from
    // parameter pack and forward rest
    template <int pos, typename ArgT, typename... Ts>
    inline llvm::Error addArgs(JITLambda::Argument *jitArgs, ArgT *arg,
                               size_t size, Ts... remainder) {
      if (auto err = jitArgs->setArg(pos, arg, size)) {
        return StreamStringError()
               << "Cannot push tensor argument " << pos << ": " << err;
      }

      return this->addArgs<pos + 1>(jitArgs, remainder...);
    }

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
