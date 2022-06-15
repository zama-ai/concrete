// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_TYPES_H_
#define CONCRETELANG_CLIENTLIB_TYPES_H_

#include <cstdint>
#include <vector>

extern "C" {
#include "concrete-ffi.h"
}

namespace concretelang {
namespace clientlib {

template <size_t N> struct MemRefDescriptor {
  uint64_t *allocated;
  uint64_t *aligned;
  size_t offset;
  size_t sizes[N];
  size_t strides[N];
};

using decrypted_scalar_t = std::uint64_t;
using decrypted_tensor_1_t = std::vector<decrypted_scalar_t>;
using decrypted_tensor_2_t = std::vector<decrypted_tensor_1_t>;
using decrypted_tensor_3_t = std::vector<decrypted_tensor_2_t>;

template <size_t Rank> using encrypted_tensor_t = MemRefDescriptor<Rank>;
using encrypted_scalar_t = uint64_t *;
using encrypted_scalars_t = uint64_t *;

struct TensorData {
  std::vector<uint64_t> values; // tensor of rank r + 1
  std::vector<int64_t> sizes;   // r sizes

  inline size_t length() {
    if (sizes.empty()) {
      return 0;
    }
    size_t len = 1;
    for (auto size : sizes) {
      len *= size;
    }
    return len;
  }

  inline size_t lweSize() { return sizes.back(); }
};

} // namespace clientlib
} // namespace concretelang
#endif
