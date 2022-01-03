// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_SERIALIZERS_ARGUMENTS_H
#define CONCRETELANG_CLIENTLIB_SERIALIZERS_ARGUMENTS_H

#include <iostream>

#include "concretelang/ClientLib/ClientParameters.h"
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

std::ostream &operator<<(std::ostream &ostream, const ClientParameters &params);
std::istream &operator>>(std::istream &istream, ClientParameters &params);

std::ostream &operator<<(std::ostream &ostream,
                         const RuntimeContext &runtimeContext);
std::istream &operator>>(std::istream &istream, RuntimeContext &runtimeContext);

std::ostream &serializeEncryptedValues(std::vector<size_t> &sizes,
                                       encrypted_scalars_t values,
                                       std::ostream &ostream);

std::ostream &
serializeEncryptedValues(encrypted_scalars_and_sizes_t &values_and_sizes,
                         std::ostream &ostream);

encrypted_scalars_and_sizes_t unserializeEncryptedValues(
    std::vector<int64_t> &expectedSizes, // includes lweSize, unsigned to
                                         // accomodate non static sizes
    std::istream &istream);

} // namespace clientlib
} // namespace concretelang

#endif
