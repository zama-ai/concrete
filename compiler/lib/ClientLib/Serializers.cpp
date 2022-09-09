// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iosfwd>
#include <iostream>
#include <stdlib.h>

#include "concrete-core-ffi.h"

#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

template <typename Engine, typename Result>
Result read_deser(std::istream &istream,
                  int (*deser)(Engine *, BufferView, Result *),
                  Engine *engine) {
  size_t length;
  readSize(istream, length);
  // buffer is too big to be allocated on stack
  // vector ensures everything is deallocated w.r.t. new
  std::vector<uint8_t> buffer(length);
  istream.read((char *)buffer.data(), length);
  assert(istream.good());
  Result result;

  CAPI_ASSERT_ERROR(deser(engine, {buffer.data(), length}, &result));

  return result;
}

template <typename BufferLike>
std::ostream &writeBufferLike(std::ostream &ostream, BufferLike &buffer) {
  writeSize(ostream, buffer.length);
  ostream.write((const char *)buffer.pointer, buffer.length);
  assert(ostream.good());
  return ostream;
}

std::ostream &operator<<(std::ostream &ostream, const LweKeyswitchKey64 *key) {
  DefaultSerializationEngine *engine;

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

  Buffer b;

  CAPI_ASSERT_ERROR(
      default_serialization_engine_serialize_lwe_keyswitch_key_u64(engine, key,
                                                                   &b));

  writeBufferLike(ostream, b);
  free((void *)b.pointer);
  b.pointer = nullptr;
  return ostream;
}

std::ostream &operator<<(std::ostream &ostream, const LweBootstrapKey64 *key) {
  DefaultSerializationEngine *engine;

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

  Buffer b;

  CAPI_ASSERT_ERROR(
      default_serialization_engine_serialize_lwe_bootstrap_key_u64(engine, key,
                                                                   &b))

  writeBufferLike(ostream, b);
  free((void *)b.pointer);
  b.pointer = nullptr;
  return ostream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const FftwFourierLweBootstrapKey64 *key) {
  FftwSerializationEngine *engine;

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(new_fftw_serialization_engine(&engine));

  Buffer b;

  CAPI_ASSERT_ERROR(
      fftw_serialization_engine_serialize_fftw_fourier_lwe_bootstrap_key_u64(
          engine, key, &b))

  writeBufferLike(ostream, b);
  free((void *)b.pointer);
  b.pointer = nullptr;
  return ostream;
}

std::istream &operator>>(std::istream &istream, LweKeyswitchKey64 *&key) {
  DefaultSerializationEngine *engine;

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

  key = read_deser(
      istream, default_serialization_engine_deserialize_lwe_keyswitch_key_u64,
      engine);
  return istream;
}

std::istream &operator>>(std::istream &istream, LweBootstrapKey64 *&key) {
  DefaultSerializationEngine *engine;

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

  key = read_deser(
      istream, default_serialization_engine_deserialize_lwe_bootstrap_key_u64,
      engine);
  return istream;
}

std::istream &operator>>(std::istream &istream,
                         FftwFourierLweBootstrapKey64 *&key) {
  FftwSerializationEngine *engine;

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(new_fftw_serialization_engine(&engine));

  key = read_deser(
      istream,
      fftw_serialization_engine_deserialize_fftw_fourier_lwe_bootstrap_key_u64,
      engine);
  return istream;
}

std::istream &operator>>(std::istream &istream,
                         RuntimeContext &runtimeContext) {
  istream >> runtimeContext.evaluationKeys;
  assert(istream.good());
  return istream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const RuntimeContext &runtimeContext) {
  ostream << runtimeContext.evaluationKeys;
  assert(ostream.good());
  return ostream;
}

template <typename T>
static std::istream &unserializeTensorDataElements(TensorData &values_and_sizes,
                                                   std::istream &istream) {
  readWords(istream, values_and_sizes.getElementPointer<T>(0),
            values_and_sizes.getNumElements());

  return istream;
}

std::ostream &serializeTensorData(const TensorData &values_and_sizes,
                                  std::ostream &ostream) {
  switch (values_and_sizes.getElementType()) {
  case ElementType::u64:
    return serializeTensorDataRaw<uint64_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<uint64_t>(), ostream);
  case ElementType::i64:
    return serializeTensorDataRaw<int64_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<int64_t>(), ostream);
  case ElementType::u32:
    return serializeTensorDataRaw<uint32_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<uint32_t>(), ostream);
  case ElementType::i32:
    return serializeTensorDataRaw<int32_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<int32_t>(), ostream);
  case ElementType::u16:
    return serializeTensorDataRaw<uint16_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<uint16_t>(), ostream);
  case ElementType::i16:
    return serializeTensorDataRaw<int16_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<int16_t>(), ostream);
  case ElementType::u8:
    return serializeTensorDataRaw<uint8_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<uint8_t>(), ostream);
  case ElementType::i8:
    return serializeTensorDataRaw<int8_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<int8_t>(), ostream);
  }

  assert(false && "Unhandled element type");
}

outcome::checked<TensorData, StringError> unserializeTensorData(
    std::vector<int64_t> &expectedSizes, // includes lweSize, unsigned to
                                         // accomodate non static sizes
    std::istream &istream) {

  if (incorrectMode(istream)) {
    return StringError("Stream is in incorrect mode");
  }

  uint64_t numDimensions;
  readWord(istream, numDimensions);

  std::vector<size_t> dims;

  for (uint64_t i = 0; i < numDimensions; i++) {
    int64_t dimSize;
    readWord(istream, dimSize);

    if (dimSize != expectedSizes[i]) {
      istream.setstate(std::ios::badbit);
      return StringError("Number of dimensions did not match the number of "
                         "expected dimensions");
    }

    dims.push_back(dimSize);
  }

  uint64_t elementWidth;
  readWord(istream, elementWidth);

  switch (elementWidth) {
  case 64:
  case 32:
  case 16:
  case 8:
    break;
  default:
    return StringError("Element width must be either 64, 32, 16 or 8, but got ")
           << elementWidth;
  }

  uint8_t elementSignedness;
  readWord(istream, elementSignedness);

  if (elementSignedness != 0 && elementSignedness != 1) {
    return StringError("Numerical value for element signedness must be either "
                       "0 or 1, but got ")
           << elementSignedness;
  }

  TensorData result(dims, elementWidth, elementSignedness == 1);

  switch (result.getElementType()) {
  case ElementType::u64:
    unserializeTensorDataElements<uint64_t>(result, istream);
    break;
  case ElementType::i64:
    unserializeTensorDataElements<int64_t>(result, istream);
    break;
  case ElementType::u32:
    unserializeTensorDataElements<uint32_t>(result, istream);
    break;
  case ElementType::i32:
    unserializeTensorDataElements<int32_t>(result, istream);
    break;
  case ElementType::u16:
    unserializeTensorDataElements<uint16_t>(result, istream);
    break;
  case ElementType::i16:
    unserializeTensorDataElements<int16_t>(result, istream);
    break;
  case ElementType::u8:
    unserializeTensorDataElements<uint8_t>(result, istream);
    break;
  case ElementType::i8:
    unserializeTensorDataElements<int8_t>(result, istream);
    break;
  }

  return std::move(result);
}

std::ostream &operator<<(std::ostream &ostream,
                         const LweKeyswitchKey &wrappedKsk) {
  ostream << wrappedKsk.ksk;
  assert(ostream.good());
  return ostream;
}
std::istream &operator>>(std::istream &istream, LweKeyswitchKey &wrappedKsk) {
  istream >> wrappedKsk.ksk;
  assert(istream.good());
  return istream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const LweBootstrapKey &wrappedBsk) {
  ostream << wrappedBsk.bsk;
  assert(ostream.good());
  return ostream;
}
std::istream &operator>>(std::istream &istream, LweBootstrapKey &wrappedBsk) {
  istream >> wrappedBsk.bsk;
  assert(istream.good());
  return istream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const EvaluationKeys &evaluationKeys) {
  bool has_ksk = (bool)evaluationKeys.sharedKsk;
  writeWord(ostream, has_ksk);
  if (has_ksk) {
    ostream << *evaluationKeys.sharedKsk;
  }

  bool has_bsk = (bool)evaluationKeys.sharedBsk;
  writeWord(ostream, has_bsk);
  if (has_bsk) {
    ostream << *evaluationKeys.sharedBsk;
  }
  assert(ostream.good());
  return ostream;
}

std::istream &operator>>(std::istream &istream,
                         EvaluationKeys &evaluationKeys) {
  bool has_ksk;
  readWord(istream, has_ksk);
  if (has_ksk) {
    auto sharedKsk = LweKeyswitchKey(nullptr);
    istream >> sharedKsk;
    evaluationKeys.sharedKsk =
        std::make_shared<LweKeyswitchKey>(std::move(sharedKsk));
  }

  bool has_bsk;
  readWord(istream, has_bsk);
  if (has_bsk) {
    auto sharedBsk = LweBootstrapKey(nullptr);
    istream >> sharedBsk;
    evaluationKeys.sharedBsk =
        std::make_shared<LweBootstrapKey>(std::move(sharedBsk));
  }

  assert(istream.good());
  return istream;
}

} // namespace clientlib
} // namespace concretelang
