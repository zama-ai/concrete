// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "lib.h"
#include "concretelang/Support/CompilerEngine.h"
#include "cxx.h"
#include <memory>

using mlir::concretelang::CompilationContext;
using mlir::concretelang::CompilerEngine;

namespace concrete_sys {

// CompilationOptions
std::unique_ptr<CompilationOptions> New() {
  return std::make_unique<CompilationOptions>(CompilationOptions{mlir::concretelang::CompilationOptions()});
}

// Library
std::unique_ptr<ProgramInfo> Library::getProgramInfo() {
  auto maybeProgramInfo = this->inner.getProgramInfo();
  if (maybeProgramInfo.has_error()) {
    throw std::runtime_error(std::move(maybeProgramInfo.error().mesg));
  }
  return std::make_unique<ProgramInfo>(ProgramInfo{maybeProgramInfo.value()});
}

rust::String Library::getStaticLibraryPath(){
  return this->inner.getStaticLibraryPath();
}

std::unique_ptr<Library> compile(rust::Str sources,
                                 const CompilationOptions &options,
                                 rust::Str outputDirPath) {
  auto cxxSources = (std::string)sources;
  auto cxxOutputDirPath = (std::string)outputDirPath;
  auto context = CompilationContext::createShared();
  auto engine = CompilerEngine(context);
  engine.setCompilationOptions(options.inner);
  std::unique_ptr<llvm::MemoryBuffer> mb =
      llvm::MemoryBuffer::getMemBuffer(cxxSources);
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(mb), llvm::SMLoc());
  auto maybeLibrary =
      engine.compile(sm, cxxOutputDirPath, "", false, true, true, true);
  if (auto err = maybeLibrary.takeError()) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }
  mlir::concretelang::Library output = *maybeLibrary;
  return std::make_unique<Library>(Library{output});
}



} // namespace concrete_sys
