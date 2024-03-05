// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_UTILS_H_
#define CONCRETELANG_SUPPORT_UTILS_H_

#include "concrete-protocol.capnp.h"
#include "concretelang/Runtime/context.h"
#include "concretelang/Support/Error.h"
#include "llvm/ADT/SmallVector.h"

namespace concretelang {

/// prefix function name with `concrete_` to avoid collision with other function
std::string prefixFuncName(llvm::StringRef funcName);

// construct the function name of the wrapper function that unify function calls
// of compiled circuit
std::string makePackedFunctionName(llvm::StringRef name);

// memref is a struct which is flattened aligned, allocated pointers, offset,
// and two array of rank size for sizes and strides.
uint64_t numArgOfRankedMemrefCallingConvention(uint64_t rank);

template <typename V, unsigned int N>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const llvm::SmallVector<V, N> vect) {
  OS << "[";
  for (auto v : vect) {
    OS << v << ",";
  }
  OS << "]";
  return OS;
}
} // namespace concretelang

#endif
