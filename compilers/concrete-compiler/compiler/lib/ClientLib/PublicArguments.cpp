// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <memory>
#include <stdlib.h>
#include <sys/types.h>
#include <variant>
#include <vector>

#ifdef OUTPUT_COMPRESSION_SUPPORT
#include "compress_lwe/defines.h"
#include "compress_lwe/library.h"
#endif
#include "concretelang/ClientLib/CRT.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/ClientLib/Types.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

// TODO: optimize the move
PublicArguments::PublicArguments(
    const ClientParameters &clientParameters,
    std::vector<clientlib::SharedScalarOrTensorOrCompressedData> &buffers)
    : clientParameters(clientParameters), arguments() {
  for (auto &a : buffers) {
    if (std::holds_alternative<TensorData>(a.get())) {
      arguments.push_back(std::get<TensorData>(std::move(a.get())));
    } else if (std::holds_alternative<ScalarData>(a.get())) {
      arguments.push_back(std::get<ScalarData>(a.get()));
    } else {
#ifdef OUTPUT_COMPRESSION_SUPPORT
      assert(
          std::holds_alternative<std::shared_ptr<comp::CompressedCiphertext>>(
              a.get()));
#endif
      exit(1);
    }
  }
}

PublicArguments::PublicArguments(
    const ClientParameters &clientParameters,
    std::vector<clientlib::ScalarOrTensorData> &&buffers)
    : clientParameters(clientParameters) {
  arguments = std::move(buffers);
}

PublicArguments::~PublicArguments() {}

outcome::checked<void, StringError>
PublicArguments::serialize(std::ostream &ostream) {
  if (incorrectMode(ostream)) {
    return StringError(
        "PublicArguments::serialize: ostream should be in binary mode");
  }
  serializeVectorOfScalarOrTensorData(arguments, ostream);
  if (ostream.bad()) {
    return StringError(
        "PublicArguments::serialize: cannot serialize public arguments");
  }
  return outcome::success();
}

outcome::checked<void, StringError>
PublicArguments::unserializeArgs(std::istream &istream) {
  OUTCOME_TRY(arguments, unserializeVectorOfScalarOrTensorData(istream));
  return outcome::success();
}

outcome::checked<std::unique_ptr<PublicArguments>, StringError>
PublicArguments::unserialize(const ClientParameters &expectedParams,
                             std::istream &istream) {
  std::vector<ScalarOrTensorData> emptyBuffers;
  auto sArguments = std::make_unique<PublicArguments>(expectedParams,
                                                      std::move(emptyBuffers));
  OUTCOME_TRYV(sArguments->unserializeArgs(istream));
  return std::move(sArguments);
}

outcome::checked<void, StringError>
PublicResult::unserialize(std::istream &istream) {
  OUTCOME_TRY(buffers,
              unserializeVectorOfScalarOrTensorDataOrCompressed(istream));
  return outcome::success();
}

outcome::checked<void, StringError>
PublicResult::serialize(std::ostream &ostream) {
  serializeVectorOfScalarOrTensorDataOrCompressed(buffers, ostream);
  if (ostream.bad()) {
    return StringError("PublicResult::serialize: cannot serialize");
  }
  return outcome::success();
}

void next_coord_index(size_t index[], size_t sizes[], size_t rank) {
  // increase multi dim index
  for (int r = rank - 1; r >= 0; r--) {
    if (index[r] < sizes[r] - 1) {
      index[r]++;
      return;
    }
    index[r] = 0;
  }
}

size_t global_index(size_t index[], size_t sizes[], size_t strides[],
                    size_t rank) {
  // compute global index from multi dim index
  size_t g_index = 0;
  size_t default_stride = 1;
  for (int r = rank - 1; r >= 0; r--) {
    g_index += index[r] * ((strides[r] == 0) ? default_stride : strides[r]);
    default_stride *= sizes[r];
  }
  return g_index;
}

static inline bool isReferenceToMLIRGlobalMemory(void *ptr) {
  return reinterpret_cast<uintptr_t>(ptr) == 0xdeadbeef;
}

template <typename T>
TensorData tensorDataFromMemRefTyped(size_t memref_rank, void *allocatedVoid,
                                     void *alignedVoid, size_t offset,
                                     size_t *sizes, size_t *strides) {
  T *allocated = reinterpret_cast<T *>(allocatedVoid);
  T *aligned = reinterpret_cast<T *>(alignedVoid);

  TensorData result(llvm::ArrayRef<size_t>{sizes, memref_rank}, sizeof(T) * 8,
                    std::is_signed<T>());
  assert(aligned != nullptr);

  // ephemeral multi dim index to compute global strides
  size_t *index = new size_t[memref_rank];
  for (size_t r = 0; r < memref_rank; r++) {
    index[r] = 0;
  }
  auto len = result.length();

  // TODO: add a fast path for dense result (no real strides)
  for (size_t i = 0; i < len; i++) {
    int g_index = offset + global_index(index, sizes, strides, memref_rank);
    result.getElementReference<T>(i) = aligned[g_index];
    next_coord_index(index, sizes, memref_rank);
  }
  delete[] index;
  // TEMPORARY: That quick and dirty but as this function is used only to
  // convert a result of the mlir program and as data are copied here, we
  // release the alocated pointer if it set.

  if (allocated != nullptr && !isReferenceToMLIRGlobalMemory(allocated)) {
    free(allocated);
  }

  return result;
}

TensorData tensorDataFromMemRef(size_t memref_rank, size_t element_width,
                                bool is_signed, void *allocated, void *aligned,
                                size_t offset, size_t *sizes, size_t *strides) {
  ElementType et = getElementTypeFromWidthAndSign(element_width, is_signed);

  switch (et) {
  case ElementType::i64:
    return tensorDataFromMemRefTyped<int64_t>(memref_rank, allocated, aligned,
                                              offset, sizes, strides);
  case ElementType::u64:
    return tensorDataFromMemRefTyped<uint64_t>(memref_rank, allocated, aligned,
                                               offset, sizes, strides);
  case ElementType::i32:
    return tensorDataFromMemRefTyped<int32_t>(memref_rank, allocated, aligned,
                                              offset, sizes, strides);
  case ElementType::u32:
    return tensorDataFromMemRefTyped<uint32_t>(memref_rank, allocated, aligned,
                                               offset, sizes, strides);
  case ElementType::i16:
    return tensorDataFromMemRefTyped<int16_t>(memref_rank, allocated, aligned,
                                              offset, sizes, strides);
  case ElementType::u16:
    return tensorDataFromMemRefTyped<uint16_t>(memref_rank, allocated, aligned,
                                               offset, sizes, strides);
  case ElementType::i8:
    return tensorDataFromMemRefTyped<int8_t>(memref_rank, allocated, aligned,
                                             offset, sizes, strides);
  case ElementType::u8:
    return tensorDataFromMemRefTyped<uint8_t>(memref_rank, allocated, aligned,
                                              offset, sizes, strides);
  }

  // Cannot happen
  assert(false);
}

outcome::checked<std::vector<uint64_t>, StringError>
decrypt_no_decode_one_result(const KeySet &keySet, uint pos, CircuitGate gate,
                             uint ct_count,
                             ScalarOrTensorOrCompressedData &buffers,
                             const ClientParameters &clientParameters) {
  if (std::holds_alternative<ScalarData>(buffers)) {
    abort();
  }

  auto encoding = gate.encryption->encoding;

  std::vector<uint64_t> decrypted_vector;

  auto &crt = encoding.crt;

  uint64_t lweSize = clientParameters.lweBufferSize(gate);
  uint64_t number_lwe_to_decrypt;
  if (crt.empty()) {
    lweSize = clientParameters.lweBufferSize(gate);
    number_lwe_to_decrypt = ct_count;
  } else {
    lweSize = clientParameters.lweBufferSize(gate) / crt.size();
    number_lwe_to_decrypt = ct_count * crt.size();
  }

  if (std::holds_alternative<TensorData>(buffers)) {

    auto &buffer = std::get<TensorData>(buffers);

    assert(lweSize * number_lwe_to_decrypt == buffer.length());

    for (size_t i = 0; i < number_lwe_to_decrypt; i++) {
      auto ciphertext = buffer.getOpaqueElementPointer(i * lweSize);

      auto ciphertextu64 = reinterpret_cast<uint64_t *>(ciphertext);
      OUTCOME_TRY(uint64_t decrypted, keySet.decrypt_lwe(pos, ciphertextu64));
      decrypted_vector.push_back(decrypted);
    }
  } else {
#ifdef OUTPUT_COMPRESSION_SUPPORT

    assert(std::holds_alternative<std::shared_ptr<comp::CompressedCiphertext>>(
        buffers));

    auto &arg = std::get<std::shared_ptr<comp::CompressedCiphertext>>(buffers);

    auto &fullKeys = keySet.getFullKey();

    assert(fullKeys.has_value());

    auto &privKey = fullKeys->ahe_sk;

    decrypted_vector =
        comp::decryptCompressedBatched(*arg, *privKey, number_lwe_to_decrypt);
#else
    exit(1);
#endif
  }
  return decrypted_vector;
}

outcome::checked<std::vector<uint64_t>, StringError>
decode_one_result(std::vector<uint64_t> decrypted_vector, Encoding encoding) {
  auto &crt = encoding.crt;

  std::vector<uint64_t> decoded_vector;

  if (crt.empty()) {

    auto precision = encoding.precision;

    for (auto decrypted : decrypted_vector) {

      decoded_vector.push_back(
          decode_1padded_integer(decrypted, precision, encoding.isSigned));
    }
  } else {
    uint crt_size = crt.size();

    assert(decrypted_vector.size() % crt_size == 0);

    uint ct_count = decrypted_vector.size() / crt_size;

    for (uint i = 0; i < ct_count; i++) {
      std::vector<int64_t> remainders;
      for (uint j = 0; j < crt_size; j++) {

        auto plaintext = concretelang::clientlib::crt::decode(
            decrypted_vector[i * crt_size + j], crt[j]);
        remainders.push_back(plaintext);
      }

      // Compute the inverse crt
      auto output = decode_crt(remainders, crt, encoding.isSigned);

      decoded_vector.push_back(output);
    }
  }

  return decoded_vector;
}

outcome::checked<std::vector<uint64_t>, StringError>
decrypt_decode_one_result(const KeySet &keySet, uint pos, CircuitGate gate,
                          uint ct_count,
                          ScalarOrTensorOrCompressedData &buffers,
                          const ClientParameters &clientParameters) {
  OUTCOME_TRY(std::vector<uint64_t> decrypted_vector,
              decrypt_no_decode_one_result(keySet, pos, gate, ct_count, buffers,
                                           clientParameters));

  return decode_one_result(decrypted_vector, gate.encryption->encoding);
}

uint64_t decode_1padded_integer(uint64_t decrypted, uint64_t precision,
                                bool signe) {

  uint64_t shifted = decrypted + ((uint64_t)1 << (64 - precision - 2));

  uint64_t decoded = shifted >> (64 - precision - 1);

  // remove padding bit
  // decoded %= 1 << precision;

  // Further decode signed integers.
  if (signe) {
    uint64_t maxPos = ((uint64_t)1 << (precision - 1));
    if (decoded >= maxPos) { // The output is actually negative.
      // Set the preceding bits to zero
      decoded |= UINT64_MAX << precision;
      // This makes sure when the value is cast to int64, it has the
      // correct value
    };
  }

  return decoded;
}

uint64_t decode_crt(std::vector<int64_t> &decrypted_remainders,
                    std::vector<int64_t> &crt_bases, bool signe) {
  // Compute the inverse crt
  auto output = crt::iCrt(crt_bases, decrypted_remainders);

  // Further decode signed integers
  if (signe) {
    uint64_t maxPos = 1;
    for (auto prime : crt_bases) {
      maxPos *= prime;
    }
    maxPos /= 2;
    if (output >= maxPos) {
      output -= maxPos * 2;
    }
  }

  return output;
}

} // namespace clientlib
} // namespace concretelang
