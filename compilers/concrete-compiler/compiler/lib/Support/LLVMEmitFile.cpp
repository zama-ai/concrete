// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <errno.h>

#include "llvm/MC/SubtargetFeature.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>

#include <mlir/Support/FileUtilities.h>

#include <concretelang/Support/Error.h>
#include <concretelang/Support/Utils.h>

namespace mlir {
namespace concretelang {

using std::string;
using std::vector;

// Get target machine from current machine and setup LLVM module accordingly
std::unique_ptr<llvm::TargetMachine>
getTargetMachineAndSetupModule(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    llvm::errs() << "NO target: " << errorMessage << "\n";
    return nullptr;
  }

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, llvm::Reloc::PIC_));
  if (!machine) {
    llvm::errs() << "Unable to create target machine\n";
    return nullptr;
  }
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
  return machine;
}

// This function was copied from the MLIR Execution Engine, and provide an
// elegant and generic invocation interface to the compiled circuit:
// For each function in the LLVM module, define an interface function that wraps
// all the arguments of the original function and all its results into an i8**
// pointer to provide a unified invocation interface.
static void packFunctionArguments(llvm::Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  llvm::DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // prefix to avoid colliding with other functions
    func.setName(::concretelang::prefixFuncName(func.getName()));

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto *newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = ::concretelang::makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc =
        llvm::cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto &indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
      llvm::Value *argPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, argIndex);
      llvm::Value *argPtr =
          builder.CreateLoad(builder.getInt8PtrTy(), argPtrPtr);
      llvm::Type *argTy = indexedArg.value().getType();
      argPtr = builder.CreateBitCast(argPtr, argTy->getPointerTo());
      llvm::Value *arg = builder.CreateLoad(argTy, argPtr);
      args.push_back(arg);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, retIndex);
      llvm::Value *retPtr =
          builder.CreateLoad(builder.getInt8PtrTy(), retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}

llvm::Error emitObject(llvm::Module &module, string objectPath) {
  auto targetMachine = getTargetMachineAndSetupModule(&module);
  if (!targetMachine) {
    return StreamStringError("No default target machine for object generation");
  }

  string Error;
  std::unique_ptr<llvm::ToolOutputFile> objectFile =
      mlir::openOutputFile(objectPath, &Error);
  if (!objectFile) {
    return StreamStringError("Cannot create/open " + objectPath);
  }

  packFunctionArguments(&module);

  // The legacy PassManager is mandatory for final code generation.
  // https://llvm.org/docs/NewPassManager.html#status-of-the-new-and-legacy-pass-managers
  llvm::legacy::PassManager pm;
  auto FileType = llvm::CGFT_ObjectFile;
  targetMachine->setOptLevel(llvm::CodeGenOpt::Level::Aggressive);
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
