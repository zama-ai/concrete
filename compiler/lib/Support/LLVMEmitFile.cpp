#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <mlir/Support/FileUtilities.h>

#include <zamalang/Support/Error.h>

namespace mlir {
namespace zamalang {

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
  auto RM = llvm::Optional<llvm::Reloc::Model>();
  return Target->createTargetMachine(TargetTriple, CPU, Features, opt, RM);
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

  return llvm::Error::success();
}

string linkerCmd(vector<string> objectsPath, string libraryPath,
                 string linker) {
  string cmd = linker + libraryPath;
  for (auto objectPath : objectsPath) {
    cmd += " " + objectPath;
  }
  cmd += " 2>&1"; // to keep stderr with popen
  return cmd;
}

llvm::Error callCmd(string cmd) {
  FILE *fp = popen(cmd.c_str(), "r");
  if (fp == NULL)
    return StreamStringError("Cannot call the linker: " + cmd);

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
                        string linker) {
  auto cmd = linkerCmd(objectsPath, libraryPath, linker);
  return callCmd(cmd);
}

} // namespace zamalang
} // namespace mlir
