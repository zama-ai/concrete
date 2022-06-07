// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_LLVMEMITFILE
#define CONCRETELANG_SUPPORT_LLVMEMITFILE

namespace mlir {
namespace concretelang {

llvm::Error emitObject(llvm::Module &module, std::string objectPath);

llvm::Error callCmd(std::string cmd);

llvm::Error
emitLibrary(std::vector<std::string> objectsPath, std::string libraryPath,
            std::string linker,
            llvm::Optional<std::vector<std::string>> extraArgs = {});

} // namespace concretelang
} // namespace mlir

#endif