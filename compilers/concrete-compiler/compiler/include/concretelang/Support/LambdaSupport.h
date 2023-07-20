// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_LAMBDASUPPORT
#define CONCRETELANG_SUPPORT_LAMBDASUPPORT

#include "boost/outcome.h"

#include "concretelang/Support/LambdaArgument.h"

#include "concretelang/ClientLib/ClientLambda.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/Common/Error.h"
#include "concretelang/ServerLib/ServerLambda.h"

namespace mlir {
namespace concretelang {

namespace clientlib = ::concretelang::clientlib;

namespace {
// Generic function template as well as specializations of
// `typedResult` must be declared at namespace scope due to return
// type template specialization

/// Helper function for implementing type-dependent preparation of the result.
template <typename ResT>
llvm::Expected<ResT> typedResult(clientlib::KeySet &keySet,
                                 clientlib::PublicResult &result);

template <typename T>
inline llvm::Expected<T> typedScalarResult(clientlib::KeySet &keySet,
                                           clientlib::PublicResult &result) {
  auto clearResult = result.asClearTextScalar<T>(keySet, 0);
  if (!clearResult.has_value()) {
    return StreamStringError("typedResult cannot get clear text scalar")
           << clearResult.error().mesg;
  }

  return clearResult.value();
}

/// Specializations of `typedResult()` for scalar results, forwarding
/// scalar value to caller.

template <>
inline llvm::Expected<uint64_t> typedResult(clientlib::KeySet &keySet,
                                            clientlib::PublicResult &result) {
  return typedScalarResult<uint64_t>(keySet, result);
}
template <>
inline llvm::Expected<int64_t> typedResult(clientlib::KeySet &keySet,
                                           clientlib::PublicResult &result) {
  return typedScalarResult<int64_t>(keySet, result);
}
template <>
inline llvm::Expected<uint32_t> typedResult(clientlib::KeySet &keySet,
                                            clientlib::PublicResult &result) {
  return typedScalarResult<uint32_t>(keySet, result);
}
template <>
inline llvm::Expected<int32_t> typedResult(clientlib::KeySet &keySet,
                                           clientlib::PublicResult &result) {
  return typedScalarResult<int32_t>(keySet, result);
}
template <>
inline llvm::Expected<uint16_t> typedResult(clientlib::KeySet &keySet,
                                            clientlib::PublicResult &result) {
  return typedScalarResult<uint16_t>(keySet, result);
}
template <>
inline llvm::Expected<int16_t> typedResult(clientlib::KeySet &keySet,
                                           clientlib::PublicResult &result) {
  return typedScalarResult<int16_t>(keySet, result);
}
template <>
inline llvm::Expected<uint8_t> typedResult(clientlib::KeySet &keySet,
                                           clientlib::PublicResult &result) {
  return typedScalarResult<uint8_t>(keySet, result);
}
template <>
inline llvm::Expected<int8_t> typedResult(clientlib::KeySet &keySet,
                                          clientlib::PublicResult &result) {
  return typedScalarResult<int8_t>(keySet, result);
}

template <typename T>
inline llvm::Expected<std::vector<T>>
typedVectorResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  auto clearResult = result.asClearTextVector<T>(keySet, 0);
  if (!clearResult.has_value()) {
    return StreamStringError("typedVectorResult cannot get clear text vector")
           << clearResult.error().mesg;
  }
  return std::move(clearResult.value());
}

/// Specializations of `typedResult()` for vector results, initializing
/// an `std::vector` of the right size with the results and forwarding
/// it to the caller with move semantics.
/// Cannot factor out into a template template <typename T> inline
/// llvm::Expected<std::vector<uint8_t>>
/// typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result); due
/// to ambiguity with scalar template
template <>
inline llvm::Expected<std::vector<uint8_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<uint8_t>(keySet, result);
}
template <>
inline llvm::Expected<std::vector<int8_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<int8_t>(keySet, result);
}
template <>
inline llvm::Expected<std::vector<uint16_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<uint16_t>(keySet, result);
}
template <>
inline llvm::Expected<std::vector<int16_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<int16_t>(keySet, result);
}
template <>
inline llvm::Expected<std::vector<uint32_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<uint32_t>(keySet, result);
}
template <>
inline llvm::Expected<std::vector<int32_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<int32_t>(keySet, result);
}
template <>
inline llvm::Expected<std::vector<uint64_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<uint64_t>(keySet, result);
}
template <>
inline llvm::Expected<std::vector<int64_t>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  return typedVectorResult<int64_t>(keySet, result);
}

template <typename T>
llvm::Expected<std::unique_ptr<LambdaArgument>>
buildTensorLambdaResult(clientlib::KeySet &keySet,
                        clientlib::PublicResult &result) {
  llvm::Expected<std::vector<T>> tensorOrError =
      typedResult<std::vector<T>>(keySet, result);
  if (auto err = tensorOrError.takeError())
    return std::move(err);

  auto tensorDim = result.asClearTextShape(0);
  if (tensorDim.has_error())
    return StreamStringError(tensorDim.error().mesg);

  return std::make_unique<TensorLambdaArgument<IntLambdaArgument<T>>>(
      *tensorOrError, tensorDim.value());
}

template <typename T>
llvm::Expected<std::unique_ptr<LambdaArgument>>
buildScalarLambdaResult(clientlib::KeySet &keySet,
                        clientlib::PublicResult &result) {
  llvm::Expected<T> scalarOrError = typedResult<T>(keySet, result);
  if (auto err = scalarOrError.takeError())
    return std::move(err);

  return std::make_unique<IntLambdaArgument<T>>(*scalarOrError);
}

/// pecialization of `typedResult()` for a single result wrapped into
/// a `LambdaArgument`.
template <>
inline llvm::Expected<std::unique_ptr<LambdaArgument>>
typedResult(clientlib::KeySet &keySet, clientlib::PublicResult &result) {
  auto gate = keySet.outputGate(0);
  auto width = gate.shape.width;
  bool sign = gate.shape.sign;

  if (width > 64)
    return StreamStringError("Cannot handle values with more than 64 bits");

  // By convention, decrypted integers are always 64 bits wide
  if (gate.isEncrypted())
    width = 64;

  if (gate.shape.dimensions.empty()) {
    // scalar case
    if (width > 32) {
      return (sign) ? buildScalarLambdaResult<int64_t>(keySet, result)
                    : buildScalarLambdaResult<uint64_t>(keySet, result);
    } else if (width > 16) {
      return (sign) ? buildScalarLambdaResult<int32_t>(keySet, result)
                    : buildScalarLambdaResult<uint32_t>(keySet, result);
    } else if (width > 8) {
      return (sign) ? buildScalarLambdaResult<int16_t>(keySet, result)
                    : buildScalarLambdaResult<uint16_t>(keySet, result);
    } else if (width <= 8) {
      return (sign) ? buildScalarLambdaResult<int8_t>(keySet, result)
                    : buildScalarLambdaResult<uint8_t>(keySet, result);
    }
  } else if (gate.chunkInfo.has_value()) {
    // chunked scalar case
    assert(gate.shape.dimensions.size() == 1);
    width = gate.shape.size * gate.chunkInfo->width;
    if (width > 32) {
      return (sign) ? buildScalarLambdaResult<int64_t>(keySet, result)
                    : buildScalarLambdaResult<uint64_t>(keySet, result);
    } else if (width > 16) {
      return (sign) ? buildScalarLambdaResult<int32_t>(keySet, result)
                    : buildScalarLambdaResult<uint32_t>(keySet, result);
    } else if (width > 8) {
      return (sign) ? buildScalarLambdaResult<int16_t>(keySet, result)
                    : buildScalarLambdaResult<uint16_t>(keySet, result);
    } else if (width <= 8) {
      return (sign) ? buildScalarLambdaResult<int8_t>(keySet, result)
                    : buildScalarLambdaResult<uint8_t>(keySet, result);
    }
  } else {
    // tensor case
    if (width > 32) {
      return (sign) ? buildTensorLambdaResult<int64_t>(keySet, result)
                    : buildTensorLambdaResult<uint64_t>(keySet, result);
    } else if (width > 16) {
      return (sign) ? buildTensorLambdaResult<int32_t>(keySet, result)
                    : buildTensorLambdaResult<uint32_t>(keySet, result);
    } else if (width > 8) {
      return (sign) ? buildTensorLambdaResult<int16_t>(keySet, result)
                    : buildTensorLambdaResult<uint16_t>(keySet, result);
    } else if (width <= 8) {
      return (sign) ? buildTensorLambdaResult<int8_t>(keySet, result)
                    : buildTensorLambdaResult<uint8_t>(keySet, result);
    }
  }

  assert(false && "Cannot happen");
}
} // namespace

/// Adaptor class that push arguments specified as instances of
/// `LambdaArgument` to `clientlib::EncryptedArguments`.
class LambdaArgumentAdaptor {
public:
  /// Checks if the argument `arg` is an plaintext / encrypted integer
  /// argument or a plaintext / encrypted tensor argument with a
  /// backing integer type `IntT` and push the argument to `encryptedArgs`.
  ///
  /// Returns `true` if `arg` has one of the types above and its value
  /// was successfully added to `encryptedArgs`, `false` if none of the types
  /// matches or an error if a type matched, but adding the argument to
  /// `encryptedArgs` failed.
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

  /// Recursive case for `tryAddArg<IntT>(...)`
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

  /// Attempts to push a single argument `arg` to `encryptedArgs`. Returns an
  /// error if either the argument type is unsupported or if the argument types
  /// is supported, but adding it to `encryptedArgs` failed.
  static inline llvm::Error
  addArgument(clientlib::EncryptedArguments &encryptedArgs,
              const LambdaArgument &arg, clientlib::KeySet &keySet) {
    // Try the supported integer types; size_t needs explicit
    // treatment, since it may alias none of the fixed size integer
    // types
    llvm::Expected<bool> successOrError =
        LambdaArgumentAdaptor::tryAddArg<int64_t, int32_t, int16_t, int8_t,
                                         uint64_t, uint32_t, uint16_t, uint8_t,
                                         size_t>(encryptedArgs, arg, keySet);

    if (!successOrError)
      return successOrError.takeError();

    if (successOrError.get() == false)
      return StreamStringError("Unknown argument type");
    else
      return llvm::Error::success();
  }

  /// Encrypts and build public arguments from lambda arguments
  static llvm::Expected<std::unique_ptr<clientlib::PublicArguments>>
  exportArguments(llvm::ArrayRef<const LambdaArgument *> args,
                  clientlib::ClientParameters clientParameters,
                  clientlib::KeySet &keySet) {

    auto encryptedArgs = clientlib::EncryptedArguments::empty();
    for (auto arg : args) {
      if (auto err = LambdaArgumentAdaptor::addArgument(*encryptedArgs, *arg,
                                                        keySet)) {
        return std::move(err);
      }
    }
    auto check = encryptedArgs->checkAllArgs(keySet);
    if (check.has_error()) {
      return StreamStringError(check.error().mesg);
    }
    auto publicArguments =
        encryptedArgs->exportPublicArguments(clientParameters);
    if (publicArguments.has_error()) {
      return StreamStringError(publicArguments.error().mesg);
    }
    return std::move(publicArguments.value());
  }
};

template <typename Lambda, typename CompilationResult> class LambdaSupport {
public:
  typedef Lambda lambda;
  typedef CompilationResult compilationResult;

  virtual ~LambdaSupport() {}

  /// Compile the mlir program and produces a compilation result if succeed.
  llvm::Expected<std::unique_ptr<CompilationResult>> virtual compile(
      llvm::SourceMgr &program,
      CompilationOptions options = CompilationOptions("main")) = 0;

  llvm::Expected<std::unique_ptr<CompilationResult>> virtual compile(
      mlir::ModuleOp program,
      std::shared_ptr<mlir::concretelang::CompilationContext> cctx,
      CompilationOptions options = CompilationOptions("main")) = 0;

  llvm::Expected<std::unique_ptr<CompilationResult>>
  compile(llvm::StringRef program,
          CompilationOptions options = CompilationOptions("main")) {
    return compile(llvm::MemoryBuffer::getMemBuffer(program), options);
  }

  llvm::Expected<std::unique_ptr<CompilationResult>>
  compile(std::unique_ptr<llvm::MemoryBuffer> program,
          CompilationOptions options = CompilationOptions("main")) {
    llvm::SourceMgr sm;
    sm.AddNewSourceBuffer(std::move(program), llvm::SMLoc());
    return compile(sm, options);
  }

  /// Load the server lambda from the compilation result.
  llvm::Expected<Lambda> virtual loadServerLambda(
      CompilationResult &result) = 0;

  /// Load the client parameters from the compilation result.
  llvm::Expected<clientlib::ClientParameters> virtual loadClientParameters(
      CompilationResult &result) = 0;

  /// Load the compilation feedback from the compilation result.
  llvm::Expected<CompilationFeedback> virtual loadCompilationFeedback(
      CompilationResult &result) = 0;

  /// Call the lambda with the public arguments.
  llvm::Expected<std::unique_ptr<clientlib::PublicResult>> virtual serverCall(
      Lambda lambda, clientlib::PublicArguments &args,
      clientlib::EvaluationKeys &evaluationKeys) = 0;

  /// Build the client KeySet from the client parameters.
  static llvm::Expected<std::unique_ptr<clientlib::KeySet>>
  keySet(clientlib::ClientParameters clientParameters,
         std::optional<clientlib::KeySetCache> cache, uint64_t seed_msb = 0,
         uint64_t seed_lsb = 0) {
    std::shared_ptr<clientlib::KeySetCache> cachePtr;
    if (cache.has_value()) {
      cachePtr = std::make_shared<clientlib::KeySetCache>(cache.value());
    }
    auto keySet = clientlib::KeySetCache::generate(cachePtr, clientParameters,
                                                   seed_msb, seed_lsb);
    if (keySet.has_error()) {
      return StreamStringError(keySet.error().mesg);
    }
    return std::move(keySet.value());
  }

  static llvm::Expected<std::unique_ptr<clientlib::PublicArguments>>
  exportArguments(clientlib::ClientParameters clientParameters,
                  clientlib::KeySet &keySet,
                  llvm::ArrayRef<const LambdaArgument *> args) {
    return LambdaArgumentAdaptor::exportArguments(args, clientParameters,
                                                  keySet);
  }

  template <typename ResT>
  static llvm::Expected<ResT> call(Lambda lambda,
                                   clientlib::PublicArguments &publicArguments,
                                   clientlib::EvaluationKeys &evaluationKeys) {
    // Call the lambda
    auto publicResult = LambdaSupport<Lambda, CompilationResult>().serverCall(
        lambda, publicArguments, evaluationKeys);
    if (auto err = publicResult.takeError()) {
      return std::move(err);
    }

    // Decrypt the result
    return typedResult<ResT>(keySet, **publicResult);
  }
};

template <class LambdaSupport> class ClientServer {
public:
  static llvm::Expected<ClientServer>
  create(llvm::StringRef program,
         CompilationOptions options = CompilationOptions("main"),
         std::optional<clientlib::KeySetCache> cache = {},
         LambdaSupport support = LambdaSupport()) {
    auto compilationResult = support.compile(program, options);
    if (auto err = compilationResult.takeError()) {
      return std::move(err);
    }
    auto lambda = support.loadServerLambda(**compilationResult);
    if (auto err = lambda.takeError()) {
      return std::move(err);
    }
    auto clientParameters = support.loadClientParameters(**compilationResult);
    if (auto err = clientParameters.takeError()) {
      return std::move(err);
    }
    auto keySet = support.keySet(*clientParameters, cache);
    if (auto err = keySet.takeError()) {
      return std::move(err);
    }
    auto f = ClientServer();
    f.lambda = *lambda;
    f.compilationResult = std::move(*compilationResult);
    f.keySet = std::move(*keySet);
    f.clientParameters = *clientParameters;
    f.support = support;
    return std::move(f);
  }

  template <typename ResT = uint64_t>
  llvm::Expected<ResT> operator()(llvm::ArrayRef<LambdaArgument *> args) {
    auto publicArguments = LambdaArgumentAdaptor::exportArguments(
        args, clientParameters, *this->keySet);

    if (auto err = publicArguments.takeError()) {
      return std::move(err);
    }

    auto evaluationKeys = this->keySet->evaluationKeys();
    auto publicResult =
        support.serverCall(lambda, **publicArguments, evaluationKeys);
    if (auto err = publicResult.takeError()) {
      return std::move(err);
    }
    return typedResult<ResT>(*keySet, **publicResult);
  }

  template <typename T, typename ResT = uint64_t>
  llvm::Expected<ResT> operator()(const llvm::ArrayRef<T> args) {
    auto encryptedArgs = clientlib::EncryptedArguments::create(
        /*simulation*/ false, *keySet, args);
    if (encryptedArgs.has_error()) {
      return StreamStringError(encryptedArgs.error().mesg);
    }
    auto publicArguments =
        encryptedArgs.value()->exportPublicArguments(clientParameters);
    if (!publicArguments.has_value()) {
      return StreamStringError(publicArguments.error().mesg);
    }
    auto evaluationKeys = keySet->evaluationKeys();
    auto publicResult =
        support.serverCall(lambda, *publicArguments.value(), evaluationKeys);
    if (auto err = publicResult.takeError()) {
      return std::move(err);
    }
    return typedResult<ResT>(*keySet, **publicResult);
  }

  template <typename ResT = uint64_t, typename... Args>
  llvm::Expected<ResT> operator()(const Args... args) {
    auto encryptedArgs = clientlib::EncryptedArguments::create(
        /*simulation*/ false, *keySet, args...);
    if (encryptedArgs.has_error()) {
      return StreamStringError(encryptedArgs.error().mesg);
    }
    auto publicArguments =
        encryptedArgs.value()->exportPublicArguments(clientParameters);
    if (publicArguments.has_error()) {
      return StreamStringError(publicArguments.error().mesg);
    }
    auto evaluationKeys = keySet->evaluationKeys();
    auto publicResult =
        support.serverCall(lambda, *publicArguments.value(), evaluationKeys);
    if (auto err = publicResult.takeError()) {
      return std::move(err);
    }
    return typedResult<ResT>(*keySet, **publicResult);
  }

private:
  typename LambdaSupport::lambda lambda;
  std::unique_ptr<typename LambdaSupport::compilationResult> compilationResult;
  std::unique_ptr<clientlib::KeySet> keySet;
  clientlib::ClientParameters clientParameters;
  LambdaSupport support;
};

} // namespace concretelang
} // namespace mlir

#endif
