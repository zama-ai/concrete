// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETE_SYS_LIB_H
#define CONCRETE_SYS_LIB_H

#include "cxx.h"
#include "concretelang/Support/CompilerEngine.h"
#include <memory>

// Contains helpers not available from the original compiler api.
namespace concrete_sys{

struct CompilationOptions{
    mlir::concretelang::CompilationOptions inner;
};

std::unique_ptr<CompilationOptions> New();

struct ProgramInfo {
    concretelang::protocol::Message<concreteprotocol::ProgramInfo> inner;
};

struct Library {
    mlir::concretelang::Library inner;
    std::unique_ptr<ProgramInfo> getProgramInfo();
    rust::String getStaticLibraryPath();
};

std::unique_ptr<Library> compile(
    rust::Str sm,
    const CompilationOptions &options,
    rust::Str outputDirPath
);

} // namespace concrete_sys

#endif
