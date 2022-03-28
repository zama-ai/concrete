#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

print(
"""// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

// generated: see genDynamicRandCall.py

#include <cassert>
#include <vector>

#include "concretelang/ClientLib/Types.h"
#include "concretelang/ServerLib/DynamicArityCall.h"
#include "concretelang/ServerLib/ServerLambda.h"

namespace concretelang {
namespace serverlib {

TensorData
multi_arity_call_dynamic_rank(void* (*func)(void *...), std::vector<void *> args, size_t rank) {
  using concretelang::clientlib::MemRefDescriptor;
  constexpr auto convert = concretelang::clientlib::tensorDataFromMemRef;
  switch (rank) {""")

for tensor_rank in range(0, 33):
    memref_rank = tensor_rank + 1
    print(f"""    case {tensor_rank}:
    {{
      auto m = multi_arity_call((MemRefDescriptor<{memref_rank}>(*)(void *...))func, args);
      return convert({memref_rank}, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
    }}""")

print("""
 default:
      assert(false);
  }
}""")

print("""
} // namespace serverlib
} // namespace concretelang""")
