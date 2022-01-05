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
  LweCiphertext_u64 **allocated;
  LweCiphertext_u64 **aligned;
  size_t offset;
  size_t sizes[N];
  size_t strides[N];
};

llvm::Expected<std::vector<uint64_t>> decryptSlice(LweCiphertext_u64 **aligned,
                                                   KeySet &keySet, size_t start,
                                                   size_t size,
                                                   size_t stride = 1) {
  stride = (stride == 0) ? 1 : stride;
  std::vector<uint64_t> result(size);
  for (size_t i = 0; i < size; i++) {
    size_t offset = start + i * stride;
    auto err = keySet.decrypt_lwe(0, aligned[offset], result[i]);
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
  lambda.func =
      (void *(*)(void *, ...))dlsym(module->libraryHandle, funcName.c_str());

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
  auto fCasted = (LweCiphertext_u64 * (*)(void *...))(lambda.func);
  ;
  LweCiphertext_u64 *lweResult =
      mlir::concretelang::call(fCasted, args.preparedArgs);

  uint64_t decryptedResult;
  if (auto err = lambda.keySet->decrypt_lwe(0, lweResult, decryptedResult)) {
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
  if (output.shape.dimensions.size() != Rank) {
    return StreamStringError("the function doesn't return a tensor of rank ")
           << Rank;
  }
  // Tensor encrypted result
  auto fCasted = (MemRefDescriptor<Rank>(*)(void *...))(func);
  auto encryptedResult = mlir::concretelang::call(fCasted, args.preparedArgs);

  for (size_t dim = 0; dim < Rank; dim++) {
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
  auto encryptedResultOrErr = lambda.invokeMemRefDecriptor<1>(args);
  if (!encryptedResultOrErr) {
    return encryptedResultOrErr.takeError();
  }
  auto &encryptedResult = encryptedResultOrErr.get();
  auto &keySet = lambda.keySet;
  return decryptSlice(encryptedResult.aligned, *keySet, encryptedResult.offset,
                      encryptedResult.sizes[0], encryptedResult.strides[0]);
}

template <>
llvm::Expected<std::vector<std::vector<uint64_t>>>
invoke<std::vector<std::vector<uint64_t>>>(DynamicLambda &lambda,
                                           const Arguments &args) {
  auto encryptedResultOrErr = lambda.invokeMemRefDecriptor<2>(args);
  if (!encryptedResultOrErr) {
    return encryptedResultOrErr.takeError();
  }
  auto &encryptedResult = encryptedResultOrErr.get();
  auto &keySet = lambda.keySet;

  std::vector<std::vector<uint64_t>> result;
  result.reserve(encryptedResult.sizes[0]);
  for (size_t i = 0; i < encryptedResult.sizes[0]; i++) {
    // TODO : strides
    int offset = encryptedResult.offset + i * encryptedResult.sizes[1];
    auto slice =
        decryptSlice(encryptedResult.aligned, *keySet, offset,
                     encryptedResult.sizes[1], encryptedResult.strides[1]);
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
  auto encryptedResultOrErr = lambda.invokeMemRefDecriptor<3>(args);
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
      // TODO : strides
      int offset = encryptedResult.offset +
                   i * encryptedResult.sizes[1] * encryptedResult.sizes[2] +
                   j * encryptedResult.sizes[2];
      auto slice =
          decryptSlice(encryptedResult.aligned, *keySet, offset,
                       encryptedResult.sizes[2], encryptedResult.strides[2]);
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