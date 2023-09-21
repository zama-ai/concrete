#include <concretelang/Analysis/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

using ::concretelang::error::StringError;

namespace mlir {
namespace concretelang {
std::string locationString(mlir::Location loc) {
  auto location = std::string();
  auto locationStream = llvm::raw_string_ostream(location);
  loc->print(locationStream);
  return location;
}
} // namespace concretelang
} // namespace mlir
