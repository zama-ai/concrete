// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <errno.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>

#include <mlir/Support/FileUtilities.h>

#include <concretelang/Support/Error.h>

namespace mlir {
namespace concretelang {

using std::string;
using std::vector;

llvm::TargetMachine *getDefaultTargetMachine() {
  auto TargetTriple = llvm::sys::getDefaultTargetTriple();
  string Error;

  auto Target = llvm::TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!Target) {
    return nullptr;
  }

  auto CPU = "generic";
  auto Features = "";
  llvm::TargetOptions opt;
  return Target->createTargetMachine(TargetTriple, CPU, Features, opt,
                                     llvm::Reloc::PIC_);
}

llvm::Error emitObject(llvm::Module &module, string objectPath) {
  auto targetMachine = getDefaultTargetMachine();
  if (!targetMachine) {
    return StreamStringError("No default target machine for object generation");
  }

  module.setDataLayout(targetMachine->createDataLayout());

  string Error;
  std::unique_ptr<llvm::ToolOutputFile> objectFile =
      mlir::openOutputFile(objectPath, &Error);
  if (!objectFile) {
    return StreamStringError("Cannot create/open " + objectPath);
  }

  // The legacy PassManager is mandatory for final code generation.
  // https://llvm.org/docs/NewPassManager.html#status-of-the-new-and-legacy-pass-managers
  llvm::legacy::PassManager pm;
  auto FileType = llvm::CGFT_ObjectFile;
  if (targetMachine->addPassesToEmitFile(pm, objectFile->os(), nullptr,
                                         FileType, false)) {
    return StreamStringError("TheTargetMachine can't emit object file");
  }

  pm.run(module);

  objectFile->os().flush();
  objectFile->os().close();
  objectFile->keep();
  delete targetMachine;
  return llvm::Error::success();
}

string linkerCmd(vector<string> objectsPath, string libraryPath, string linker,
                 std::optional<vector<string>> extraArgs) {
  string cmd = linker + libraryPath;
  for (auto objectPath : objectsPath) {
    cmd += " " + objectPath;
  }
  if (extraArgs.has_value()) {
    for (auto extraArg : extraArgs.value()) {
      cmd += " " + extraArg;
    }
  }
  cmd += " 2>&1"; // to keep stderr with popen
  return cmd;
}

llvm::Error callCmd(string cmd) {
  errno = 0;
  FILE *fp = popen(cmd.c_str(), "r");
  if (fp == NULL) {
    return StreamStringError(strerror(errno))
           << "\nCannot call the linker: " << cmd;
  }

  string outputContent;
  const int CHUNK_SIZE = 1024;
  char chunk[CHUNK_SIZE];

  while (fgets(chunk, CHUNK_SIZE, fp) != NULL) {
    outputContent += chunk;
  }

  int status = pclose(fp);

  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    return llvm::Error::success();
  } else if (status == -1) {
    return StreamStringError("Cannot pclose: " + cmd);
  } else {
    return StreamStringError("Command failed:" + cmd + "\nCode:" +
                             std::to_string(status) + "\n" + outputContent);
  }
}

llvm::Error emitLibrary(vector<string> objectsPath, string libraryPath,
                        string linker,
                        std::optional<vector<string>> extraArgs) {
  auto cmd = linkerCmd(objectsPath, libraryPath, linker, extraArgs);
  return callCmd(cmd);
}

} // namespace concretelang
} // namespace mlir
