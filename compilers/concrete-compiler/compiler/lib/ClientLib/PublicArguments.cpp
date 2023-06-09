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
    std::vector<clientlib::SharedScalarOrTensorData> &buffers)
    : clientParameters(clientParameters) {
  arguments = buffers;
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
  std::vector<SharedScalarOrTensorData> emptyBuffers;
  auto sArguments =
      std::make_unique<PublicArguments>(expectedParams, emptyBuffers);
  OUTCOME_TRYV(sArguments->unserializeArgs(istream));
  return std::move(sArguments);
}

outcome::checked<void, StringError>
PublicResult::unserialize(std::istream &istream) {
  OUTCOME_TRY(buffers, unserializeVectorOfScalarOrTensorData(istream));
  return outcome::success();
}

outcome::checked<void, StringError>
PublicResult::serialize(std::ostream &ostream) {
  serializeVectorOfScalarOrTensorData(buffers, ostream);
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

} // namespace clientlib
} // namespace concretelang
