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
