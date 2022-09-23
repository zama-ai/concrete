// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_SERIALIZERS_ARGUMENTS_H
#define CONCRETELANG_CLIENTLIB_SERIALIZERS_ARGUMENTS_H

#include <iostream>
#include <limits>

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Runtime/context.h"

namespace concretelang {
namespace clientlib {

using RuntimeContext = mlir::concretelang::RuntimeContext;

// integers are not serialized as binary values even on a binary stream
// so we cannot rely on << operator directly
template <typename Word>
std::ostream &writeWord(std::ostream &ostream, Word word) {
  ostream.write(reinterpret_cast<char *>(&(word)), sizeof(word));
  assert(ostream.good());
  return ostream;
}

template <typename Size>
std::ostream &writeSize(std::ostream &ostream, Size size) {
  return writeWord(ostream, size);
}

// for sake of symetry
template <typename Word>
std::istream &readWord(std::istream &istream, Word &word) {
  istream.read(reinterpret_cast<char *>(&(word)), sizeof(word));
  assert(istream.good());
  return istream;
}

template <typename Word>
std::istream &readWords(std::istream &istream, Word *words, size_t numWords) {
  assert(std::numeric_limits<size_t>::max() / sizeof(*words) > numWords);
  istream.read(reinterpret_cast<char *>(words), sizeof(*words) * numWords);
  assert(istream.good());
  return istream;
}

template <typename Size>
std::istream &readSize(std::istream &istream, Size &size) {
  return readWord(istream, size);
}

template <typename Stream> bool incorrectMode(Stream &stream) {
  auto binary = stream.flags() && std::ios::binary;
  if (!binary) {
    stream.setstate(std::ios::failbit);
  }
  return !binary;
}

std::ostream &operator<<(std::ostream &ostream,
                         const RuntimeContext &runtimeContext);
std::istream &operator>>(std::istream &istream, RuntimeContext &runtimeContext);

std::ostream &serializeScalarData(const ScalarData &sd, std::ostream &ostream);

outcome::checked<ScalarData, StringError>
unserializeScalarData(std::istream &istream);

std::ostream &serializeTensorData(const TensorData &values_and_sizes,
                                  std::ostream &ostream);

template <typename T>
std::ostream &serializeTensorDataRaw(const llvm::ArrayRef<size_t> &dimensions,
                                     const llvm::ArrayRef<T> &values,
                                     std::ostream &ostream) {

  writeWord<uint64_t>(ostream, dimensions.size());

  for (size_t dim : dimensions)
    writeWord<int64_t>(ostream, dim);

  writeWord<uint64_t>(ostream, sizeof(T) * 8);
  writeWord<uint8_t>(ostream, std::is_signed<T>());

  for (T val : values)
    writeWord(ostream, val);

  return ostream;
}

outcome::checked<TensorData, StringError> unserializeTensorData(
    std::vector<int64_t> &expectedSizes, // includes unsigned to
                                         // accomodate non static sizes
    std::istream &istream);

std::ostream &serializeScalarOrTensorData(const ScalarOrTensorData &sotd,
                                          std::ostream &ostream);

outcome::checked<ScalarOrTensorData, StringError>
unserializeScalarOrTensorData(const std::vector<int64_t> &expectedSizes,
                              std::istream &istream);

std::ostream &operator<<(std::ostream &ostream,
                         const LweKeyswitchKey &wrappedKsk);
std::istream &operator>>(std::istream &istream, LweKeyswitchKey &wrappedKsk);

std::ostream &operator<<(std::ostream &ostream,
                         const LweBootstrapKey &wrappedBsk);
std::istream &operator>>(std::istream &istream, LweBootstrapKey &wrappedBsk);

std::ostream &operator<<(std::ostream &ostream,
                         const EvaluationKeys &evaluationKeys);
std::istream &operator>>(std::istream &istream, EvaluationKeys &evaluationKeys);

} // namespace clientlib
} // namespace concretelang

#endif
