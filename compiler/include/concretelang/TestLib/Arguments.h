// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TESTLIB_ARGUMENTS_H
#define CONCRETELANG_TESTLIB_ARGUMENTS_H

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/KeySet.h"

namespace mlir {
namespace concretelang {

class DynamicLambda;

class Arguments {
public:
  Arguments(KeySet &keySet) : currentPos(0), keySet(keySet) {
    keySet.setRuntimeContext(context);
  }

  ~Arguments();

  // Create EncryptedArgument that use the given KeySet to perform encryption
  // and decryption operations.
  static std::shared_ptr<Arguments> create(KeySet &keySet);

  // Add a scalar argument.
  llvm::Error pushArg(uint64_t arg);

  // Add a vector-tensor argument.
  llvm::Error pushArg(std::vector<uint8_t> arg);

  template <size_t size> llvm::Error pushArg(std::array<uint8_t, size> arg) {
    return pushArg(8, (void *)arg.data(), {size});
  }

  // Add a matrix-tensor argument.
  template <size_t size0, size_t size1>
  llvm::Error pushArg(std::array<std::array<uint8_t, size1>, size0> arg) {
    return pushArg(8, (void *)arg.data(), {size0, size1});
  }

  // Add a rank3 tensor.
  template <size_t size0, size_t size1, size_t size2>
  llvm::Error pushArg(
      std::array<std::array<std::array<uint8_t, size2>, size1>, size0> arg) {
    return pushArg(8, (void *)arg.data(), {size0, size1, size2});
  }

  // Generalize by computing shape by template recursion

  // Set a argument at the given pos as a 1D tensor of T.
  template <typename T> llvm::Error pushArg(T *data, int64_t dim1) {
    return pushArg<T>(data, llvm::ArrayRef<int64_t>(&dim1, 1));
  }

  // Set a argument at the given pos as a tensor of T.
  template <typename T>
  llvm::Error pushArg(T *data, llvm::ArrayRef<int64_t> shape) {
    return pushArg(8 * sizeof(T), static_cast<void *>(data), shape);
  }

  llvm::Error pushArg(size_t width, void *data, llvm::ArrayRef<int64_t> shape);

  // Push the runtime context to the argument list, this must be called
  // after each argument was pushed.
  llvm::Error pushContext();

  template <typename Arg0, typename... OtherArgs>
  llvm::Error pushArgs(Arg0 arg0, OtherArgs... others) {
    auto err = pushArg(arg0);
    if (err) {
      return err;
    }
    return pushArgs(others...);
  }

  llvm::Error pushArgs() { return pushContext(); }

private:
  friend DynamicLambda;
  template <typename Result>
  friend llvm::Expected<Result> invoke(DynamicLambda &lambda,
                                       const Arguments &args);
  llvm::Error checkPushTooManyArgs();

  // Position of the next pushed argument
  size_t currentPos;
  std::vector<void *> preparedArgs;

  // Store allocated lwe ciphertexts (for free)
  std::vector<LweCiphertext_u64 *> allocatedCiphertexts;
  // Store buffers of ciphertexts
  std::vector<LweCiphertext_u64 **> ciphertextBuffers;

  KeySet &keySet;
  RuntimeContext context;
};

} // namespace concretelang
} // namespace mlir

#endif
