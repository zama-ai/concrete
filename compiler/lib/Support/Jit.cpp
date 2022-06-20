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
#include <concretelang/Runtime/DFRuntime.hpp>
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Jit.h>
#include <concretelang/Support/logging.h>

namespace mlir {
namespace concretelang {

llvm::Expected<std::unique_ptr<JITLambda>>
JITLambda::create(llvm::StringRef name, mlir::ModuleOp &module,
                  llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline,
                  llvm::Optional<std::string> runtimeLibPath) {

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
  if (runtimeLibPath.hasValue())
    sharedLibPaths.push_back(runtimeLibPath.getValue());

  mlir::ExecutionEngineOptions execOptions;
  execOptions.transformer = optPipeline;
  execOptions.sharedLibPaths = sharedLibPaths;
  execOptions.jitCodeGenOptLevel = llvm::None;
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

// memref is a struct which is flattened aligned, allocated pointers, offset,
// and two array of rank size for sizes and strides.
uint64_t numArgOfRankedMemrefCallingConvention(uint64_t rank) {
  return 3 + 2 * rank;
}

llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
JITLambda::call(clientlib::PublicArguments &args,
                clientlib::EvaluationKeys &evaluationKeys) {
#ifndef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
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
    std::vector<clientlib::TensorData> buffers;
    return clientlib::PublicResult::fromBuffers(args.clientParameters, buffers);
  }
#endif

  // invokeRaw needs to have pointers on arguments and a pointers on the result
  // as last argument.
  // Prepare the outputs vector to store the output value of the lambda.
  auto numOutputs = 0;
  for (auto &output : args.clientParameters.outputs) {
    auto shape = args.clientParameters.bufferShape(output);
    if (shape.size() == 0) {
      // scalar gate
      numOutputs += 1;
    } else {
      // buffer gate
      numOutputs += numArgOfRankedMemrefCallingConvention(shape.size());
    }
  }
  std::vector<void *> outputs(numOutputs);
  // Prepare the raw arguments of invokeRaw, i.e. a vector with pointer on
  // inputs and outputs.
  std::vector<void *> rawArgs(args.preparedArgs.size() + 1 /*runtime context*/ +
                              outputs.size());
  size_t i = 0;
  // Pointers on inputs
  for (auto &arg : args.preparedArgs) {
    rawArgs[i++] = &arg;
  }

  RuntimeContext runtimeContext;
  runtimeContext.evaluationKeys = evaluationKeys;
  // Pointer on runtime context, the rawArgs take pointer on actual value that
  // is passed to the compiled function.
  auto rtCtxPtr = &runtimeContext;
  rawArgs[i++] = &rtCtxPtr;
  // Pointers on outputs
  for (auto &out : outputs) {
    rawArgs[i++] = &out;
  }
  // Invoke
  if (auto err = invokeRaw(rawArgs)) {
    return std::move(err);
  }

  // Store the result to the PublicResult
  std::vector<clientlib::TensorData> buffers;
  {
    size_t outputOffset = 0;
    for (auto &output : args.clientParameters.outputs) {
      auto shape = args.clientParameters.bufferShape(output);
      if (shape.size() == 0) {
        // scalar scalar
        buffers.push_back(
            clientlib::tensorDataFromScalar((uint64_t)outputs[outputOffset++]));
      } else {
        // buffer gate
        auto rank = shape.size();
        auto allocated = (uint64_t *)outputs[outputOffset++];
        auto aligned = (uint64_t *)outputs[outputOffset++];
        auto offset = (size_t)outputs[outputOffset++];
        size_t *sizes = (size_t *)&outputs[outputOffset];
        outputOffset += rank;
        size_t *strides = (size_t *)&outputs[outputOffset];
        outputOffset += rank;
        buffers.push_back(clientlib::tensorDataFromMemRef(
            rank, allocated, aligned, offset, sizes, strides));
      }
    }
  }
  return clientlib::PublicResult::fromBuffers(args.clientParameters, buffers);
}

} // namespace concretelang
} // namespace mlir
