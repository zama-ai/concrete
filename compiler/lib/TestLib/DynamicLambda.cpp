// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <dlfcn.h>

#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include "concretelang/TestLib/DynamicLambda.h"
#include "concretelang/TestLib/dynamicArityCall.h"

namespace mlir {
namespace concretelang {

template <size_t N> struct MemRefDescriptor {
  uint64_t *allocated;
  uint64_t *aligned;
  size_t offset;
  size_t sizes[N];
  size_t strides[N];
};

llvm::Expected<std::vector<uint64_t>>
decryptSlice(KeySet &keySet, uint64_t *aligned, size_t size) {
  auto pos = 0;
  std::vector<uint64_t> result(size);
  auto lweSize = keySet.getInputLweSecretKeyParam(pos).size + 1;
  for (size_t i = 0; i < size; i++) {
    size_t offset = i * lweSize;
    auto err = keySet.decrypt_lwe(pos, aligned + offset, result[i]);
    if (err) {
      return StreamStringError()
             << "cannot decrypt result #" << i << ", err:" << err;
    }
  }

  return result;
}

llvm::Expected<mlir::concretelang::DynamicLambda>
DynamicLambda::load(std::string funcName, std::string outputLib) {
  auto moduleOrErr = mlir::concretelang::DynamicModule::open(outputLib);
  if (!moduleOrErr) {
    return moduleOrErr.takeError();
  }
  return mlir::concretelang::DynamicLambda::load(*moduleOrErr, funcName);
}

llvm::Expected<DynamicLambda>
DynamicLambda::load(std::shared_ptr<DynamicModule> module,
                    std::string funcName) {
  DynamicLambda lambda;
  lambda.module =
      module; // prevent module and library handler from being destroyed
  lambda.func = dlsym(module->libraryHandle, funcName.c_str());

  if (auto err = dlerror()) {
    return StreamStringError("Cannot open lambda: ") << err;
  }

  auto param =
      llvm::find_if(module->clientParametersList, [&](ClientParameters param) {
        return param.functionName == funcName;
      });

  if (param == module->clientParametersList.end()) {
    return StreamStringError("cannot find function ")
           << funcName << "in client parameters";
  }

  if (param->outputs.size() != 1) {
    return StreamStringError("DynamicLambda: output arity (")
           << std::to_string(param->outputs.size())
           << ") != 1 is not supported";
  }

  if (!param->outputs[0].encryption.hasValue()) {
    return StreamStringError(
        "DynamicLambda: clear output is not yet supported");
  }

  lambda.clientParameters = *param;
  return lambda;
}

template <>
llvm::Expected<uint64_t> invoke<uint64_t>(DynamicLambda &lambda,
                                          const Arguments &args) {
  auto output = lambda.clientParameters.outputs[0];
  if (output.shape.size != 0) {
    return StreamStringError("the function doesn't return a scalar");
  }
  // Scalar encrypted result
  auto fCasted = (MemRefDescriptor<1>(*)(void *...))(lambda.func);
  MemRefDescriptor<1> lweResult =
      mlir::concretelang::call(fCasted, args.preparedArgs);

  uint64_t decryptedResult;
  if (auto err =
          lambda.keySet->decrypt_lwe(0, lweResult.aligned, decryptedResult)) {
    return std::move(err);
  }
  return decryptedResult;
}

template <size_t Rank>
llvm::Expected<MemRefDescriptor<Rank>>
DynamicLambda::invokeMemRefDecriptor(const Arguments &args) {
  auto output = clientParameters.outputs[0];
  if (output.shape.size == 0) {
    return StreamStringError("the function doesn't return a tensor");
  }
  if (output.shape.dimensions.size() != Rank - 1) {
    return StreamStringError("the function doesn't return a tensor of rank ")
           << Rank - 1;
  }
  // Tensor encrypted result
  auto fCasted = (MemRefDescriptor<Rank>(*)(void *...))(func);
  auto encryptedResult = mlir::concretelang::call(fCasted, args.preparedArgs);

  for (size_t dim = 0; dim < Rank - 1; dim++) {
    size_t actual_size = encryptedResult.sizes[dim];
    size_t expected_size = output.shape.dimensions[dim];
    if (actual_size != expected_size) {
      return StreamStringError("the function returned a vector of size ")
             << actual_size << " instead of size " << expected_size;
    }
  }
  return encryptedResult;
}

template <>
llvm::Expected<std::vector<uint64_t>>
invoke<std::vector<uint64_t>>(DynamicLambda &lambda, const Arguments &args) {
  auto encryptedResultOrErr = lambda.invokeMemRefDecriptor<2>(args);
  if (!encryptedResultOrErr) {
    return encryptedResultOrErr.takeError();
  }
  auto &encryptedResult = encryptedResultOrErr.get();
  auto &keySet = lambda.keySet;
  return decryptSlice(*keySet, encryptedResult.aligned,
                      encryptedResult.sizes[0]);
}

template <>
llvm::Expected<std::vector<std::vector<uint64_t>>>
invoke<std::vector<std::vector<uint64_t>>>(DynamicLambda &lambda,
                                           const Arguments &args) {
  auto encryptedResultOrErr = lambda.invokeMemRefDecriptor<3>(args);
  if (!encryptedResultOrErr) {
    return encryptedResultOrErr.takeError();
  }
  auto &encryptedResult = encryptedResultOrErr.get();

  std::vector<std::vector<uint64_t>> result;
  result.reserve(encryptedResult.sizes[0]);
  for (size_t i = 0; i < encryptedResult.sizes[0]; i++) {
    int offset = encryptedResult.offset + i * encryptedResult.strides[1];
    auto slice = decryptSlice(*lambda.keySet, encryptedResult.aligned + offset,
                              encryptedResult.sizes[1]);
    if (!slice) {
      return StreamStringError(llvm::toString(slice.takeError()));
    }
    result.push_back(slice.get());
  }
  return result;
}

template <>
llvm::Expected<std::vector<std::vector<std::vector<uint64_t>>>>
invoke<std::vector<std::vector<std::vector<uint64_t>>>>(DynamicLambda &lambda,
                                                        const Arguments &args) {
  auto encryptedResultOrErr = lambda.invokeMemRefDecriptor<4>(args);
  if (!encryptedResultOrErr) {
    return encryptedResultOrErr.takeError();
  }
  auto &encryptedResult = encryptedResultOrErr.get();
  auto &keySet = lambda.keySet;

  std::vector<std::vector<std::vector<uint64_t>>> result0;
  result0.reserve(encryptedResult.sizes[0]);
  for (size_t i = 0; i < encryptedResult.sizes[0]; i++) {
    std::vector<std::vector<uint64_t>> result1;
    result1.reserve(encryptedResult.sizes[1]);
    for (size_t j = 0; j < encryptedResult.sizes[1]; j++) {
      int offset = encryptedResult.offset + (i * encryptedResult.sizes[1] + j) *
                                                encryptedResult.strides[1];
      auto slice = decryptSlice(*keySet, encryptedResult.aligned + offset,
                                encryptedResult.sizes[2]);
      if (!slice) {
        return StreamStringError(llvm::toString(slice.takeError()));
      }
      result1.push_back(slice.get());
    }
    result0.push_back(result1);
  }
  return result0;
}

llvm::Error DynamicLambda::generateKeySet(llvm::Optional<KeySetCache> cache,
                                          uint64_t seed_msb,
                                          uint64_t seed_lsb) {
  auto maybeKeySet =
      cache.hasValue()
          ? cache->tryLoadOrGenerateSave(clientParameters, seed_msb, seed_lsb)
          : KeySet::generate(clientParameters, seed_msb, seed_lsb);

  if (auto err = maybeKeySet.takeError()) {
    return err;
  }
  keySet = std::move(maybeKeySet.get());
  return llvm::Error::success();
}

} // namespace concretelang
} // namespace mlir