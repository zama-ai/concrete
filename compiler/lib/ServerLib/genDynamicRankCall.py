#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

print(
"""// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

// generated: see genDynamicRandCall.py

#include <cassert>
#include <vector>

#include "concretelang/ClientLib/Types.h"
#include "concretelang/ServerLib/DynamicArityCall.h"
#include "concretelang/ServerLib/ServerLambda.h"

namespace concretelang {
namespace serverlib {

/// Helper class template that yields an unsigned integer type given a
/// size in bytes
template <std::size_t size> struct int_type_of_size {};
template <> struct int_type_of_size<4> { typedef uint32_t type; };
template <> struct int_type_of_size<8> { typedef uint64_t type; };

/// Converts one function pointer into another
// TODO: Not sure this is valid in all implementations / on all
// architectures
template <typename FnDstT, typename FnSrcT> FnDstT convert_fnptr(FnSrcT src) {
  static_assert(sizeof(FnDstT) == sizeof(FnSrcT),
                "Size of function types must match");
  using inttype = typename int_type_of_size<sizeof(FnDstT)>::type;
  inttype raw = reinterpret_cast<inttype>(src);
  return reinterpret_cast<FnDstT>(raw);
}

TensorData multi_arity_call_dynamic_rank(void *(*func)(void *...),
                                         std::vector<void *> args, size_t rank,
                                         size_t element_width, bool is_signed) {
  using concretelang::clientlib::MemRefDescriptor;
  constexpr auto convert = concretelang::clientlib::tensorDataFromMemRef;
  switch (rank) {""")

for tensor_rank in range(1, 33):
    memref_rank = tensor_rank
    print(f"""  case {tensor_rank}: {{
    auto m = multi_arity_call(
        convert_fnptr<MemRefDescriptor<{memref_rank}> (*)(void *...)>(func), args);
    return convert({memref_rank}, element_width, is_signed, m.allocated, m.aligned,
                   m.offset, m.sizes, m.strides);
  }}""")

print("""
  default:
    assert(false);
  }
}""")

print("""
} // namespace serverlib
} // namespace concretelang""")
