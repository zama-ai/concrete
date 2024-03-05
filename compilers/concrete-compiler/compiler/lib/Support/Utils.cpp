// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Support/Utils.h>

namespace concretelang {

std::string prefixFuncName(llvm::StringRef funcName) {
  return "concrete_" + funcName.str();
}

std::string makePackedFunctionName(llvm::StringRef name) {
  return "_mlir_" + name.str();
}

uint64_t numArgOfRankedMemrefCallingConvention(uint64_t rank) {
  return 3 + 2 * rank;
}

} // namespace concretelang
