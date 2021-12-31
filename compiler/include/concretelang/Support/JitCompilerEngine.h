// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_SUPPORT_JIT_COMPILER_ENGINE_H
#define CONCRETELANG_SUPPORT_JIT_COMPILER_ENGINE_H

#include "concretelang/Support/KeySetCache.h"
#include <concretelang/Support/CompilerEngine.h>
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Jit.h>
#include <concretelang/Support/LambdaArgument.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir {
namespace concretelang {

namespace {
// Generic function template as well as specializations of
// `typedResult` must be declared at namespace scope due to return
// type template specialization

// Helper function for `JitCompilerEngine::Lambda::operator()`
// implementing type-dependent preparation of the result.
template <typename ResT>
llvm::Expected<ResT> typedResult(JITLambda::Argument &arguments);

// Specialization of `typedResult()` for scalar results, forwarding
// scalar value to caller
template <>
inline llvm::Expected<uint64_t> typedResult(JITLambda::Argument &arguments) {
  uint64_t res = 0;

  if (auto err = arguments.getResult(0, res))
    return StreamStringError() << "Cannot retrieve result:" << err;

  return res;
}

template <typename T>
inline llvm::Expected<std::vector<T>>
typedVectorResult(JITLambda::Argument &arguments) {
  llvm::Expected<size_t> n = arguments.getResultVectorSize(0);

  if (auto err = n.takeError())
    return std::move(err);

  std::vector<T> res(*n);

  if (auto err = arguments.getResult(0, res.data(), res.size()))
    return StreamStringError() << "Cannot retrieve result:" << err;

  return std::move(res);
}

// Specializations of `typedResult()` for vector results, initializing
// an `std::vector` of the right size with the results and forwarding
// it to the caller with move semantics.
//
// Cannot factor out into a template template <typename T> inline
// llvm::Expected<std::vector<uint8_t>>
// typedResult(JITLambda::Argument &arguments); due to ambiguity with
// scalar template
template <>
inline llvm::Expected<std::vector<uint8_t>>
typedResult(JITLambda::Argument &arguments) {
  return std::move(typedVectorResult<uint8_t>(arguments));
}
template <>
inline llvm::Expected<std::vector<uint16_t>>
typedResult(JITLambda::Argument &arguments) {
  return std::move(typedVectorResult<uint16_t>(arguments));
}
template <>
inline llvm::Expected<std::vector<uint32_t>>
typedResult(JITLambda::Argument &arguments) {
  return std::move(typedVectorResult<uint32_t>(arguments));
}
template <>
inline llvm::Expected<std::vector<uint64_t>>
typedResult(JITLambda::Argument &arguments) {
  return std::move(typedVectorResult<uint64_t>(arguments));
}

template <typename T>
llvm::Expected<std::unique_ptr<LambdaArgument>>
buildTensorLambdaResult(JITLambda::Argument &arguments) {
  llvm::Expected<std::vector<T>> tensorOrError =
      typedResult<std::vector<T>>(arguments);

  if (!tensorOrError)
    return std::move(tensorOrError.takeError());

  llvm::Expected<std::vector<int64_t>> tensorDimOrError =
      arguments.getResultDimensions(0);

  if (!tensorDimOrError)
    return std::move(tensorDimOrError.takeError());

  return std::move(std::make_unique<TensorLambdaArgument<IntLambdaArgument<T>>>(
      *tensorOrError, *tensorDimOrError));
}

// Specialization of `typedResult()` for a single result wrapped into
// a `LambdaArgument`.
template <>
inline llvm::Expected<std::unique_ptr<LambdaArgument>>
typedResult(JITLambda::Argument &arguments) {
  llvm::Expected<enum JITLambda::Argument::ResultType> resTy =
      arguments.getResultType(0);

  if (!resTy)
    return std::move(resTy.takeError());

  switch (*resTy) {
  case JITLambda::Argument::ResultType::SCALAR: {
    uint64_t res;

    if (llvm::Error err = arguments.getResult(0, res))
      return std::move(err);

    return std::move(std::make_unique<IntLambdaArgument<uint64_t>>(res));
  }

  case JITLambda::Argument::ResultType::TENSOR: {
    llvm::Expected<size_t> width = arguments.getResultWidth(0);

    if (!width)
      return std::move(width.takeError());

    if (*width > 64)
      return StreamStringError("Cannot handle scalars with more than 64 bits");
    if (*width > 32)
      return buildTensorLambdaResult<uint64_t>(arguments);
    else if (*width > 16)
      return buildTensorLambdaResult<uint32_t>(arguments);
    else if (*width > 8)
      return buildTensorLambdaResult<uint16_t>(arguments);
    else if (*width <= 8)
      return buildTensorLambdaResult<uint8_t>(arguments);
  }
  }

  return StreamStringError("Unknown result type");
} // namespace

// Adaptor class that adds arguments specified as instances of
// `LambdaArgument` to `JitLambda::Argument`.
class JITLambdaArgumentAdaptor {
public:
  // Checks if the argument `arg` is an plaintext / encrypted integer
  // argument or a plaintext / encrypted tensor argument with a
  // backing integer type `IntT` and adds the argument to `jla` at
  // position `pos`.
  //
  // Returns `true` if `arg` has one of the types above and its value
  // was successfully added to `jla`, `false` if none of the types
  // matches or an error if a type matched, but adding the argument to
  // `jla` failed.
  template <typename IntT>
  static inline llvm::Expected<bool>
  tryAddArg(JITLambda::Argument &jla, size_t pos, const LambdaArgument &arg) {
    if (auto ila = arg.dyn_cast<IntLambdaArgument<IntT>>()) {
      if (llvm::Error err = jla.setArg(pos, ila->getValue()))
        return std::move(err);
      else
        return true;
    } else if (auto tla = arg.dyn_cast<
                          TensorLambdaArgument<IntLambdaArgument<IntT>>>()) {
      if (llvm::Error err =
              jla.setArg(pos, tla->getValue(), tla->getDimensions()))
        return std::move(err);
      else
        return true;
    }

    return false;
  }

  // Recursive case for `tryAddArg<IntT>(...)`
  template <typename IntT, typename NextIntT, typename... IntTs>
  static inline llvm::Expected<bool>
  tryAddArg(JITLambda::Argument &jla, size_t pos, const LambdaArgument &arg) {
    llvm::Expected<bool> successOrError = tryAddArg<IntT>(jla, pos, arg);

    if (!successOrError)
      return std::move(successOrError.takeError());

    if (successOrError.get() == false)
      return tryAddArg<NextIntT, IntTs...>(jla, pos, arg);
    else
      return true;
  }

  // Attempts to add a single argument `arg` to `jla` at position
  // `pos`. Returns an error if either the argument type is
  // unsupported or if the argument types is supported, but adding it
  // to `jla` failed.
  static inline llvm::Error addArgument(JITLambda::Argument &jla, size_t pos,
                                        const LambdaArgument &arg) {
    llvm::Expected<bool> successOrError =
        JITLambdaArgumentAdaptor::tryAddArg<uint64_t, uint32_t, uint16_t,
                                            uint8_t>(jla, pos, arg);

    if (!successOrError)
      return std::move(successOrError.takeError());

    if (successOrError.get() == false)
      return StreamStringError("Unknown argument type");
    else
      return llvm::Error::success();
  }
};
} // namespace

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
          compilationContext(other.compilationContext) {}

    Lambda(std::shared_ptr<CompilationContext> compilationContext,
           std::unique_ptr<JITLambda> lambda, std::unique_ptr<KeySet> keySet)
        : innerLambda(std::move(lambda)), keySet(std::move(keySet)),
          compilationContext(compilationContext) {}

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
      // Create the arguments of the JIT lambda
      llvm::Expected<std::unique_ptr<JITLambda::Argument>> argsOrErr =
          mlir::concretelang::JITLambda::Argument::create(*this->keySet.get());

      if (llvm::Error err = argsOrErr.takeError())
        return StreamStringError("Could not create lambda arguments");

      // Set the arguments
      std::unique_ptr<JITLambda::Argument> arguments =
          std::move(argsOrErr.get());

      for (size_t i = 0; i < lambdaArgs.size(); i++) {
        if (llvm::Error err = JITLambdaArgumentAdaptor::addArgument(
                *arguments, i, *lambdaArgs[i])) {
          return std::move(err);
        }
      }

      // Invoke the lambda
      if (auto err = this->innerLambda->invoke(*arguments))
        return StreamStringError() << "Cannot invoke lambda:" << err;

      return std::move(typedResult<ResT>(*arguments));
    }

    // Invocation with an array of arguments of the same type
    template <typename T, typename ResT = uint64_t>
    llvm::Expected<ResT> operator()(const llvm::ArrayRef<T> args) {
      // Create the arguments of the JIT lambda
      llvm::Expected<std::unique_ptr<JITLambda::Argument>> argsOrErr =
          mlir::concretelang::JITLambda::Argument::create(*this->keySet.get());

      if (llvm::Error err = argsOrErr.takeError())
        return StreamStringError("Could not create lambda arguments");

      // Set the arguments
      std::unique_ptr<JITLambda::Argument> arguments =
          std::move(argsOrErr.get());

      for (size_t i = 0; i < args.size(); i++) {
        if (auto err = arguments->setArg(i, args[i])) {
          return StreamStringError()
                 << "Cannot push argument " << i << ": " << err;
        }
      }

      // Invoke the lambda
      if (auto err = this->innerLambda->invoke(*arguments))
        return StreamStringError() << "Cannot invoke lambda:" << err;

      return std::move(typedResult<ResT>(*arguments));
    }

    // Invocation with arguments of different types
    template <typename ResT = uint64_t, typename... Ts>
    llvm::Expected<ResT> operator()(const Ts... ts) {
      // Create the arguments of the JIT lambda
      llvm::Expected<std::unique_ptr<JITLambda::Argument>> argsOrErr =
          mlir::concretelang::JITLambda::Argument::create(*this->keySet.get());

      if (llvm::Error err = argsOrErr.takeError())
        return StreamStringError("Could not create lambda arguments");

      // Set the arguments
      std::unique_ptr<JITLambda::Argument> arguments =
          std::move(argsOrErr.get());

      if (llvm::Error err = this->addArgs<0>(arguments.get(), ts...))
        return std::move(err);

      // Invoke the lambda
      if (auto err = this->innerLambda->invoke(*arguments))
        return StreamStringError() << "Cannot invoke lambda:" << err;

      return std::move(typedResult<ResT>(*arguments));
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
