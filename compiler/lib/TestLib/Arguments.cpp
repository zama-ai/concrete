// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang/TestLib/Arguments.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/Jit.h"

namespace mlir {
namespace concretelang {

Arguments::~Arguments() {
  for (auto ct : allocatedCiphertexts) {
    free(ct);
  }
  for (auto ctBuffer : ciphertextBuffers) {
    free(ctBuffer);
  }
}

std::shared_ptr<Arguments> Arguments::create(KeySet &keySet) {
  auto args = std::make_shared<Arguments>(keySet);
  return args;
}

llvm::Error Arguments::pushArg(uint64_t arg) {
  if (auto err = checkPushTooManyArgs()) {
    return err;
  }

  auto pos = currentPos++;
  CircuitGate input = keySet.inputGate(pos);
  if (input.shape.size != 0) {
    return StreamStringError("argument #") << pos << " is not a scalar";
  }
  if (!input.encryption.hasValue()) {
    // clear scalar: just push the argument
    if (input.shape.width != 64) {
      return StreamStringError(
          "scalar argument of with != 64 is not supported for DynamicLambda");
    }
    preparedArgs.push_back((void *)arg);
    return llvm::Error::success();
  }
  // encrypted scalar: allocate, encrypt and push
  uint64_t *ctArg;
  uint64_t ctSize = 0;
  if (auto err = keySet.allocate_lwe(pos, &ctArg, ctSize)) {
    return err;
  }
  allocatedCiphertexts.push_back(ctArg);
  if (auto err = keySet.encrypt_lwe(pos, ctArg, arg)) {
    return err;
  }
  // Note: Since we bufferized lwe ciphertext take care of memref calling
  // convention
  // allocated
  preparedArgs.push_back(nullptr);
  // aligned
  preparedArgs.push_back(ctArg);
  // offset
  preparedArgs.push_back((void *)0);
  // size
  preparedArgs.push_back((void *)ctSize);
  // stride
  preparedArgs.push_back((void *)1);

  return llvm::Error::success();
}

llvm::Error Arguments::pushArg(std::vector<uint8_t> arg) {
  return pushArg(8, (void *)arg.data(), {(int64_t)arg.size()});
}

llvm::Error Arguments::pushArg(size_t width, void *data,
                               llvm::ArrayRef<int64_t> shape) {
  if (auto err = checkPushTooManyArgs()) {
    return err;
  }
  auto pos = currentPos;
  currentPos = currentPos + 1;
  CircuitGate input = keySet.inputGate(pos);
  // Check the width of data
  if (input.shape.width > 64) {
    return StreamStringError("argument #")
           << pos << " width > 64 bits is not supported";
  }
  auto roundedSize = bitWidthAsWord(input.shape.width);
  if (width != roundedSize) {
    return StreamStringError("argument #")
           << pos << "width mismatch, got " << width << " expected "
           << roundedSize;
  }
  // Check the shape of tensor
  if (input.shape.dimensions.empty()) {
    return StreamStringError("argument #") << pos << "is not a tensor";
  }
  if (shape.size() != input.shape.dimensions.size()) {
    return StreamStringError("argument #")
           << pos << "has not the expected number of dimension, got "
           << shape.size() << " expected " << input.shape.dimensions.size();
  }
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] != input.shape.dimensions[i]) {
      return StreamStringError("argument #")
             << pos << " has not the expected dimension #" << i << " , got "
             << shape[i] << " expected " << input.shape.dimensions[i];
    }
  }
  if (input.encryption.hasValue()) {
    // Encrypted tensor: for now we support only 8 bits for encrypted tensor
    if (width != 8) {
      return StreamStringError("argument #")
             << pos << " width mismatch, expected 8 got " << width;
    }
    const uint8_t *data8 = (const uint8_t *)data;

    // Allocate a buffer for ciphertexts of size of tensor
    auto lweSize = keySet.getInputLweSecretKeyParam(pos).size + 1;
    auto ctBuffer =
        (uint64_t *)malloc(input.shape.size * lweSize * sizeof(uint64_t));
    ciphertextBuffers.push_back(ctBuffer);
    // Allocate ciphertexts and encrypt, for every values in tensor
    for (size_t i = 0, offset = 0; i < input.shape.size;
         i++, offset += lweSize) {

      if (auto err =
              this->keySet.encrypt_lwe(pos, ctBuffer + offset, data8[i])) {
        return err;
      }
    }
    // Replace the data by the buffer to ciphertext
    data = (void *)ctBuffer;
  }
  // allocated
  preparedArgs.push_back(nullptr);
  // aligned
  preparedArgs.push_back(data);
  // offset
  preparedArgs.push_back((void *)0);
  // sizes
  for (size_t i = 0; i < shape.size(); i++) {
    preparedArgs.push_back((void *)shape[i]);
  }
  // If encrypted +1 for the lwe size rank
  if (keySet.isInputEncrypted(pos)) {
    preparedArgs.push_back(
        (void *)(keySet.getInputLweSecretKeyParam(pos).size + 1));
  }
  // Set the stride for each dimension, equal to the product of the
  // following dimensions.
  int64_t stride = 1;
  // If encrypted +1 set the stride for the lwe size rank
  if (keySet.isInputEncrypted(pos)) {
    stride *= keySet.getInputLweSecretKeyParam(pos).size + 1;
  }
  for (ssize_t i = shape.size() - 1; i >= 0; i--) {
    preparedArgs.push_back((void *)stride);
    stride *= shape[i];
  }
  if (keySet.isInputEncrypted(pos)) {
    preparedArgs.push_back((void *)1);
  }
  return llvm::Error::success();
}

llvm::Error Arguments::pushContext() {
  if (currentPos < keySet.numInputs()) {
    return StreamStringError("Missing arguments");
  }
  preparedArgs.push_back(&context);
  return llvm::Error::success();
}

llvm::Error Arguments::checkPushTooManyArgs() {
  size_t arity = keySet.numInputs();
  if (currentPos < arity) {
    return llvm::Error::success();
  }
  return StreamStringError("function has arity ")
         << arity << " but is applied to too many arguments";
}

} // namespace concretelang
} // namespace mlir