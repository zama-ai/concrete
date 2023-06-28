// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_SERIALIZERS_ARGUMENTS_H
#define CONCRETELANG_CLIENTLIB_SERIALIZERS_ARGUMENTS_H

#include <iostream>
#include <limits>
#ifdef OUTPUT_COMPRESSION_SUPPORT
#include "compress_lwe/defines.h"
#endif
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/Types.h"

namespace concretelang {
namespace clientlib {

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

outcome::checked<TensorData, StringError>
unserializeTensorData(std::istream &istream);

std::ostream &serializeScalarOrTensorData(const ScalarOrTensorData &sotd,
                                          std::ostream &ostream);

outcome::checked<ScalarOrTensorData, StringError>
unserializeScalarOrTensorData(std::istream &istream);

std::ostream &serializeScalarOrTensorDataOrCompressed(
    const ScalarOrTensorOrCompressedData &sotd, std::ostream &ostream);

outcome::checked<ScalarOrTensorOrCompressedData, StringError>
unserializeScalarOrTensorDataOrCompressed(std::istream &istream);

std::ostream &
serializeVectorOfScalarOrTensorData(const std::vector<ScalarOrTensorData> &sotd,
                                    std::ostream &ostream);
outcome::checked<std::vector<ScalarOrTensorData>, StringError>
unserializeVectorOfScalarOrTensorData(std::istream &istream);

std::ostream &serializeVectorOfScalarOrTensorDataOrCompressed(
    const std::vector<SharedScalarOrTensorOrCompressedData> &sotd,
    std::ostream &ostream);

outcome::checked<std::vector<SharedScalarOrTensorOrCompressedData>, StringError>
unserializeVectorOfScalarOrTensorDataOrCompressed(std::istream &istream);

std::ostream &operator<<(std::ostream &ostream, const LweSecretKey &wrappedKsk);
LweSecretKey readLweSecretKey(std::istream &istream);

std::ostream &operator<<(std::ostream &ostream,
                         const LweKeyswitchKey &wrappedKsk);
LweKeyswitchKey readLweKeyswitchKey(std::istream &istream);

std::ostream &operator<<(std::ostream &ostream,
                         const LweBootstrapKey &wrappedBsk);
LweBootstrapKey readLweBootstrapKey(std::istream &istream);

std::ostream &operator<<(std::ostream &ostream,
                         const PackingKeyswitchKey &wrappedKsk);
PackingKeyswitchKey readPackingKeyswitchKey(std::istream &istream);

#ifdef OUTPUT_COMPRESSION_SUPPORT
std::ostream &operator<<(std::ostream &ostream, const comp::FullKeys &key);

comp::FullKeys readFullKey(std::istream &istream);
#endif

std::ostream &operator<<(std::ostream &ostream, const KeySet &keySet);
std::unique_ptr<KeySet> readKeySet(std::istream &istream);

std::ostream &operator<<(std::ostream &ostream,
                         const EvaluationKeys &evaluationKeys);
EvaluationKeys readEvaluationKeys(std::istream &istream);

} // namespace clientlib
} // namespace concretelang

#endif
