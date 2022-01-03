#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

print(
"""// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt
// for license information.

// generated: see genDynamicArityCall.py

#ifndef CONCRETELANG_SERVERLIB_DYNAMIC_ARITY_CALL_H
#define CONCRETELANG_SERVERLIB_DYNAMIC_ARITY_CALL_H


#include <cassert>
#include <vector>

#include "concretelang/ClientLib/Types.h"

namespace mlir {
namespace serverlib {


template <typename Res>
Res multi_arity_call(Res (*func)(void *...), std::vector<void *> args) {
  switch (args.size()) {
  // TODO C17++: https://en.cppreference.com/w/cpp/utility/apply
""")

for i in range(1, 128):
    args = ','.join(f'args[{j}]' for j in range(i))
    print(f'        case {i}: return func({args});')
print("""
        default:
            assert(false);
  }
}""")

print("""
} // namespace concretelang
} // namespace mlir

#endif
""")
