// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "llvm/Support/Error.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/TargetSelect.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include "concretelang/Common/BitsSize.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/Jit.h"
#include "concretelang/Support/logging.h"
#include <concretelang/Support/Utils.h>

namespace mlir {
namespace concretelang {

llvm::Expected<std::unique_ptr<JITLambda>>
JITLambda::create(llvm::StringRef name, mlir::ModuleOp &module,
                  llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline,
                  std::optional<std::string> runtimeLibPath) {

  // Looking for the function
  auto rangeOps = module.getOps<mlir::LLVM::LLVMFuncOp>();
  auto funcOp = llvm::find_if(rangeOps, [&](mlir::LLVM::LLVMFuncOp op) {
    return op.getName() == name;
  });
  if (funcOp == rangeOps.end()) {
    return llvm::make_error<llvm::StringError>(
        "cannot find the function to JIT", llvm::inconvertibleErrorCode());
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Create an MLIR execution engine. The execution engine eagerly
  // JIT-compiles the module. If runtimeLibPath is specified, it's passed as a
  // shared library to the JIT compiler.
  std::vector<llvm::StringRef> sharedLibPaths;
  if (runtimeLibPath.has_value())
    sharedLibPaths.push_back(runtimeLibPath.value());

  mlir::ExecutionEngineOptions execOptions;
  execOptions.transformer = optPipeline;
  execOptions.sharedLibPaths = sharedLibPaths;
  execOptions.jitCodeGenOptLevel = std::nullopt;
  execOptions.llvmModuleBuilder = nullptr;

  auto maybeEngine = mlir::ExecutionEngine::create(module, execOptions);
  if (!maybeEngine) {
    return StreamStringError("failed to construct the MLIR ExecutionEngine");
  }
  auto &engine = maybeEngine.get();
  auto lambda = std::make_unique<JITLambda>((*funcOp).getFunctionType(), name);
  lambda->engine = std::move(engine);

  return std::move(lambda);
}

llvm::Error JITLambda::invokeRaw(llvm::MutableArrayRef<void *> args) {
  auto found = std::find(args.begin(), args.end(), nullptr);
  if (found == args.end()) {
    return this->engine->invokePacked(this->name, args);
  }
  int pos = found - args.begin();
  return StreamStringError("invoke: argument at pos ")
         << pos << " is null or missing";
}

llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
JITLambda::call(clientlib::PublicArguments &args,
                clientlib::EvaluationKeys &evaluationKeys) {
#ifndef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED
  if (this->useDataflow) {
    return StreamStringError(
        "call: current runtime doesn't support dataflow execution, while "
        "compilation used dataflow parallelization");
  }
#else
  dfr::_dfr_set_jit(true);
  // When using JIT on distributed systems, the compiler only
  // generates work-functions and their registration calls. No results
  // are returned and no inputs are needed.
  if (!dfr::_dfr_is_root_node()) {
    std::vector<void *> rawArgs;
    if (auto err = invokeRaw(rawArgs)) {
      return std::move(err);
    }
    std::vector<clientlib::SharedScalarOrTensorData> buffers;
    return clientlib::PublicResult::fromBuffers(args.clientParameters,
                                                std::move(buffers));
  }
#endif

  return ::concretelang::invokeRawOnLambda(this, args, evaluationKeys);
}

} // namespace concretelang
} // namespace mlir
