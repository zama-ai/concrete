// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_SUPPORT_LLVMEMITFILE
#define ZAMALANG_SUPPORT_LLVMEMITFILE

namespace mlir {
namespace zamalang {

llvm::Error emitObject(llvm::Module &module, std::string objectPath);

llvm::Error emitLibrary(std::vector<std::string> objectsPath,
                        std::string libraryPath, std::string linker);

} // namespace zamalang
} // namespace mlir

#endif