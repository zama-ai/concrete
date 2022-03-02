// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/EncryptedArguments.h"
#include "concretelang/ClientLib/PublicArguments.h"

namespace concretelang {
namespace clientlib {

using StringError = concretelang::error::StringError;

outcome::checked<std::unique_ptr<PublicArguments>, StringError>
EncryptedArguments::exportPublicArguments(ClientParameters clientParameters,
                                          RuntimeContext runtimeContext) {
  // On client side the runtimeContext is hold by the KeySet
  bool clearContext = false;
  return std::make_unique<PublicArguments>(
      clientParameters, runtimeContext, clearContext, std::move(preparedArgs),
      std::move(ciphertextBuffers));
}

outcome::checked<void, StringError>
EncryptedArguments::pushArg(uint64_t arg, KeySet &keySet) {
  OUTCOME_TRYV(checkPushTooManyArgs(keySet));
  auto pos = currentPos++;
  CircuitGate input = keySet.inputGate(pos);
  if (input.shape.size != 0) {
    return StringError("argument #") << pos << " is not a scalar";
  }
  if (!input.encryption.hasValue()) {
    // clear scalar: just push the argument
    preparedArgs.push_back((void *)arg);
    return outcome::success();
  }
  ciphertextBuffers.resize(ciphertextBuffers.size() + 1); // Allocate empty
  TensorData &values_and_sizes = ciphertextBuffers.back();
  auto lweSize = keySet.getInputLweSecretKeyParam(pos).lweSize();
  values_and_sizes.sizes.push_back(lweSize);
  values_and_sizes.values.resize(lweSize);

  OUTCOME_TRYV(keySet.encrypt_lwe(pos, values_and_sizes.values.data(), arg));
  // Note: Since we bufferized lwe ciphertext take care of memref calling
  // convention
  // allocated
  preparedArgs.push_back(nullptr);
  // aligned
  preparedArgs.push_back((void *)values_and_sizes.values.data());
  // offset
  preparedArgs.push_back((void *)0);
  // size
  preparedArgs.push_back((void *)values_and_sizes.values.size());
  // stride
  preparedArgs.push_back((void *)1);
  return outcome::success();
}

outcome::checked<void, StringError>
EncryptedArguments::pushArg(std::vector<uint8_t> arg, KeySet &keySet) {
  return pushArg(8, (void *)arg.data(), {(int64_t)arg.size()}, keySet);
}

outcome::checked<void, StringError>
EncryptedArguments::pushArg(size_t width, const void *data,
                            llvm::ArrayRef<int64_t> shape, KeySet &keySet) {
  OUTCOME_TRYV(checkPushTooManyArgs(keySet));
  auto pos = currentPos;
  CircuitGate input = keySet.inputGate(pos);
  // Check the width of data
  if (input.shape.width > 64) {
    return StringError("argument #")
           << pos << " width > 64 bits is not supported";
  }
  auto roundedSize = concretelang::common::bitWidthAsWord(input.shape.width);
  if (width != roundedSize) {
    return StringError("argument #") << pos << "width mismatch, got " << width
                                     << " expected " << roundedSize;
  }
  // Check the shape of tensor
  if (input.shape.dimensions.empty()) {
    return StringError("argument #") << pos << "is not a tensor";
  }
  if (shape.size() != input.shape.dimensions.size()) {
    return StringError("argument #")
           << pos << "has not the expected number of dimension, got "
           << shape.size() << " expected " << input.shape.dimensions.size();
  }
  ciphertextBuffers.resize(ciphertextBuffers.size() + 1); // Allocate empty
  TensorData &values_and_sizes = ciphertextBuffers.back();
  for (size_t i = 0; i < shape.size(); i++) {
    values_and_sizes.sizes.push_back(shape[i]);
    if (shape[i] != input.shape.dimensions[i]) {
      return StringError("argument #")
             << pos << " has not the expected dimension #" << i << " , got "
             << shape[i] << " expected " << input.shape.dimensions[i];
    }
  }
  if (input.encryption.hasValue()) {
    auto lweSize = keySet.getInputLweSecretKeyParam(pos).lweSize();
    values_and_sizes.sizes.push_back(lweSize);

    // Encrypted tensor: for now we support only 8 bits for encrypted tensor
    if (width != 8) {
      return StringError("argument #")
             << pos << " width mismatch, expected 8 got " << width;
    }
    const uint8_t *data8 = (const uint8_t *)data;

    // Allocate a buffer for ciphertexts of size of tensor
    values_and_sizes.values.resize(input.shape.size * lweSize);
    auto &values = values_and_sizes.values;
    // Allocate ciphertexts and encrypt, for every values in tensor
    for (size_t i = 0, offset = 0; i < input.shape.size;
         i++, offset += lweSize) {
      OUTCOME_TRYV(keySet.encrypt_lwe(pos, values.data() + offset, data8[i]));
    }
  } else {
    values_and_sizes.values.resize(input.shape.size);
    for (size_t i = 0; i < input.shape.size; i++) {
      values_and_sizes.values[i] = ((const uint64_t *)data)[i];
    }
  }
  // allocated
  preparedArgs.push_back(nullptr);
  // aligned
  preparedArgs.push_back((void *)values_and_sizes.values.data());
  // offset
  preparedArgs.push_back((void *)0);
  // sizes
  for (size_t size : values_and_sizes.sizes) {
    preparedArgs.push_back((void *)size);
  }
  // Set the stride for each dimension, equal to the product of the
  // following dimensions.
  int64_t stride = values_and_sizes.length();
  // If encrypted +1 set the stride for the lwe size rank
  for (size_t size : values_and_sizes.sizes) {
    stride /= size;
    preparedArgs.push_back((void *)stride);
  }
  currentPos++;
  return outcome::success();
}

outcome::checked<void, StringError>
EncryptedArguments::checkPushTooManyArgs(KeySet &keySet) {
  size_t arity = keySet.numInputs();
  if (currentPos < arity) {
    return outcome::success();
  }
  return StringError("function has arity ")
         << arity << " but is applied to too many arguments";
}

outcome::checked<void, StringError>
EncryptedArguments::checkAllArgs(KeySet &keySet) {
  size_t arity = keySet.numInputs();
  if (currentPos == arity) {
    return outcome::success();
  }
  return StringError("function expects ")
         << arity << " arguments but has been called with " << currentPos
         << " arguments";
}

} // namespace clientlib
} // namespace concretelang
