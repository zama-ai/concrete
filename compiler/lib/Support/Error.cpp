// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include <zamalang/Support/Error.h>

namespace mlir {
namespace zamalang {
// Specialized `operator<<` for `llvm::Error` that marks the error
// as checked through `std::move` and `llvm::toString`
StreamStringError &operator<<(StreamStringError &se, llvm::Error &err) {
  se << llvm::toString(std::move(err));
  return se;
}
} // namespace zamalang
} // namespace mlir
