// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TESTLIB_DYNAMIC_LAMBDA_H
#define CONCRETELANG_TESTLIB_DYNAMIC_LAMBDA_H

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/TestLib/Arguments.h"
#include "concretelang/TestLib/DynamicModule.h"

namespace mlir {
namespace concretelang {

template <size_t N> struct MemRefDescriptor;

template <typename Result>
llvm::Expected<Result> invoke(DynamicLambda &lambda, const Arguments &args) {
  // compile time error if used
  using COMPATIBLE_RESULT_TYPE = void;
  return (Result)(COMPATIBLE_RESULT_TYPE)0; // invoke does not accept this kind
                                            // of Result
}

template <>
llvm::Expected<u_int64_t> invoke<u_int64_t>(DynamicLambda &lambda,
                                            const Arguments &args);

template <>
llvm::Expected<std::vector<uint64_t>>
invoke<std::vector<uint64_t>>(DynamicLambda &lambda, const Arguments &args);

template <>
llvm::Expected<std::vector<std::vector<uint64_t>>>
invoke<std::vector<std::vector<uint64_t>>>(DynamicLambda &lambda,
                                           const Arguments &args);

template <>
llvm::Expected<std::vector<std::vector<std::vector<uint64_t>>>>
invoke<std::vector<std::vector<std::vector<uint64_t>>>>(DynamicLambda &lambda,
                                                        const Arguments &args);

class DynamicLambda {
private:
  template <typename... Args>
  llvm::Expected<std::shared_ptr<Arguments>> createArguments(Args... args) {
    if (keySet == nullptr) {
      return StreamStringError("keySet was not initialized");
    }
    auto arg = Arguments::create(*keySet);
    auto err = arg->pushArgs(args...);
    if (err) {
      return StreamStringError(llvm::toString(std::move(err)));
    }
    return arg;
  }

public:
  static llvm::Expected<DynamicLambda> load(std::string funcName,
                                            std::string outputLib);

  static llvm::Expected<DynamicLambda>
  load(std::shared_ptr<DynamicModule> module, std::string funcName);

  template <typename Result, typename... Args>
  llvm::Expected<Result> call(Args... args) {
    auto argOrErr = createArguments(args...);
    if (!argOrErr) {
      return argOrErr.takeError();
    }
    auto arg = argOrErr.get();
    return invoke<Result>(*this, *arg);
  }

  llvm::Error generateKeySet(llvm::Optional<KeySetCache> cache = llvm::None,
                             uint64_t seed_msb = 0, uint64_t seed_lsb = 0);

protected:
  template <typename Result>
  friend llvm::Expected<Result> invoke(DynamicLambda &lambda,
                                       const Arguments &args);

  template <size_t Rank>
  llvm::Expected<MemRefDescriptor<Rank>>
  invokeMemRefDecriptor(const Arguments &args);

  ClientParameters clientParameters;
  std::shared_ptr<KeySet> keySet;
  void *(*func)(void *...);
  // Retain module and open shared lib alive
  std::shared_ptr<DynamicModule> module;
};

template <typename Result, typename... Args>
class TypedDynamicLambda : public DynamicLambda {

public:
  static llvm::Expected<TypedDynamicLambda<Result, Args...>>
  load(std::string funcName, std::string outputLib) {
    auto lambda = DynamicLambda::load(funcName, outputLib);
    if (!lambda) {
      return lambda.takeError();
    }
    return TypedDynamicLambda(*lambda);
  }

  llvm::Expected<Result> call(Args... args) {
    return DynamicLambda::call<Result>(args...);
  }

  // TODO: check parameter types
  TypedDynamicLambda(DynamicLambda &lambda) : DynamicLambda(lambda) {
    // TODO: add static check on types vs lambda inputs/outpus
  }
};

} // namespace concretelang
} // namespace mlir

#endif
