// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <stdlib.h>

#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Serializers.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

// TODO: optimize the move
PublicArguments::PublicArguments(
    const ClientParameters &clientParameters,
    std::vector<void *> &&preparedArgs_,
    std::vector<ScalarOrTensorData> &&ciphertextBuffers_)
    : clientParameters(clientParameters) {
  preparedArgs = std::move(preparedArgs_);
  ciphertextBuffers = std::move(ciphertextBuffers_);
}

PublicArguments::~PublicArguments() {}

outcome::checked<void, StringError>
PublicArguments::serialize(std::ostream &ostream) {
  if (incorrectMode(ostream)) {
    return StringError(
        "PublicArguments::serialize: ostream should be in binary mode");
  }
  size_t iPreparedArgs = 0;
  int iGate = -1;
  for (auto gate : clientParameters.inputs) {
    iGate++;
    size_t rank = gate.shape.dimensions.size();
    if (!gate.encryption.has_value()) {
      return StringError("PublicArguments::serialize: Clear arguments "
                         "are not yet supported. Argument ")
             << iGate;
    }

    /*auto allocated = */ iPreparedArgs++;
    auto aligned = (encrypted_scalars_t)preparedArgs[iPreparedArgs++];
    assert(aligned != nullptr);
    auto offset = (size_t)preparedArgs[iPreparedArgs++];
    std::vector<size_t> sizes; // includes lweSize as last dim
    sizes.resize(rank + (gate.encryption->encoding.crt.empty() ? 1 : 2));
    for (auto dim = 0u; dim < sizes.size(); dim++) {
      // sizes are part of the client parameters signature
      // it's static now but some day it could be dynamic so we serialize
      // them.
      sizes[dim] = (size_t)preparedArgs[iPreparedArgs++];
    }
    std::vector<size_t> strides(rank + 1);
    /* strides should be zero here and are not serialized */
    for (auto dim = 0u; dim < strides.size(); dim++) {
      strides[dim] = (size_t)preparedArgs[iPreparedArgs++];
    }
    // TODO: STRIDES
    auto values = aligned + offset;

    writeWord<uint8_t>(ostream, 1);
    serializeTensorDataRaw(sizes,
                           llvm::ArrayRef<clientlib::EncryptedScalarElement>{
                               values, TensorData::getNumElements(sizes)},
                           ostream);
  }

  return outcome::success();
}

outcome::checked<void, StringError>
PublicArguments::unserializeArgs(std::istream &istream) {
  int iGate = -1;
  for (auto gate : clientParameters.inputs) {
    iGate++;
    if (!gate.encryption.has_value()) {
      return StringError("Clear values are not handled");
    }

    std::vector<int64_t> sizes = gate.shape.dimensions;
    if (gate.encryption.has_value() && !gate.encryption->encoding.crt.empty()) {
      sizes.push_back(gate.encryption->encoding.crt.size());
    }
    auto lweSize = clientParameters.lweSecretKeyParam(gate).value().lweSize();
    sizes.push_back(lweSize);

    auto sotdOrErr = unserializeScalarOrTensorData(sizes, istream);

    if (sotdOrErr.has_error())
      return sotdOrErr.error();

    ciphertextBuffers.push_back(std::move(sotdOrErr.value()));
    auto &buffer = ciphertextBuffers.back();

    if (istream.fail()) {
      return StringError(
                 "PublicArguments::unserializeArgs: Failed to read argument ")
             << iGate;
    }

    if (buffer.isTensor()) {
      TensorData &td = buffer.getTensor();
      preparedArgs.push_back(/*allocated*/ nullptr);
      preparedArgs.push_back(td.getValuesAsOpaquePointer());
      preparedArgs.push_back(/*offset*/ 0);
      // sizes
      for (auto size : td.getDimensions()) {
        preparedArgs.push_back((void *)size);
      }
      // strides has been removed by serialization
      auto stride = td.length();
      for (auto size : sizes) {
        stride /= size;
        preparedArgs.push_back((void *)stride);
      }
    } else {
      ScalarData &sd = buffer.getScalar();
      preparedArgs.push_back((void *)sd.getValueAsU64());
    }
  }
  return outcome::success();
}

outcome::checked<std::unique_ptr<PublicArguments>, StringError>
PublicArguments::unserialize(ClientParameters &clientParameters,
                             std::istream &istream) {
  std::vector<void *> empty;
  std::vector<ScalarOrTensorData> emptyBuffers;
  auto sArguments = std::make_unique<PublicArguments>(
      clientParameters, std::move(empty), std::move(emptyBuffers));
  OUTCOME_TRYV(sArguments->unserializeArgs(istream));
  return std::move(sArguments);
}

outcome::checked<void, StringError>
PublicResult::unserialize(std::istream &istream) {
  for (auto gate : clientParameters.outputs) {
    if (!gate.encryption.has_value()) {
      return StringError("Clear values are not handled");
    }

    std::vector<int64_t> sizes = gate.shape.dimensions;
    if (gate.encryption.has_value() && !gate.encryption->encoding.crt.empty()) {
      sizes.push_back(gate.encryption->encoding.crt.size());
    }
    auto lweSize = clientParameters.lweSecretKeyParam(gate).value().lweSize();
    sizes.push_back(lweSize);

    auto sotd = unserializeScalarOrTensorData(sizes, istream);

    if (sotd.has_error())
      return sotd.error();

    buffers.push_back(std::move(sotd.value()));
  }
  return outcome::success();
}

outcome::checked<void, StringError>
PublicResult::serialize(std::ostream &ostream) {
  if (incorrectMode(ostream)) {
    return StringError(
        "PublicResult::serialize: ostream should be in binary mode");
  }
  for (const ScalarOrTensorData &sotd : buffers) {
    serializeScalarOrTensorData(sotd, ostream);
    if (ostream.fail()) {
      return StringError("Cannot write data");
    }
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
  default:
    // Cannot happen
    assert(false);
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
}

} // namespace clientlib
} // namespace concretelang
