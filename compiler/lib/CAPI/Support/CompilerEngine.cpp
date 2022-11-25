// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang-c/Support/CompilerEngine.h"
#include "concretelang/CAPI/Wrappers.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/LambdaSupport.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SourceMgr.h"

/// ********** CompilationOptions CAPI *****************************************

CompilationOptions
compilationOptionsCreate(MlirStringRef funcName, bool autoParallelize,
                         bool batchConcreteOps, bool dataflowParallelize,
                         bool emitGPUOps, bool loopParallelize,
                         bool optimizeConcrete, OptimizerConfig optimizerConfig,
                         bool verifyDiagnostics) {
  std::string funcNameStr(funcName.data, funcName.length);
  auto options = new mlir::concretelang::CompilationOptions(funcNameStr);
  options->autoParallelize = autoParallelize;
  options->batchConcreteOps = batchConcreteOps;
  options->dataflowParallelize = dataflowParallelize;
  options->emitGPUOps = emitGPUOps;
  options->loopParallelize = loopParallelize;
  options->optimizeConcrete = optimizeConcrete;
  options->optimizerConfig = *unwrap(optimizerConfig);
  options->verifyDiagnostics = verifyDiagnostics;
  return wrap(options);
}

CompilationOptions compilationOptionsCreateDefault() {
  return wrap(new mlir::concretelang::CompilationOptions("main"));
}

/// ********** OptimizerConfig CAPI ********************************************

OptimizerConfig optimizerConfigCreate(bool display,
                                      double fallback_log_norm_woppbs,
                                      double global_p_error, double p_error,
                                      uint64_t security, bool strategy_v0,
                                      bool use_gpu_constraints) {
  auto config = new mlir::concretelang::optimizer::Config();
  config->display = display;
  config->fallback_log_norm_woppbs = fallback_log_norm_woppbs;
  config->global_p_error = global_p_error;
  config->p_error = p_error;
  config->security = security;
  config->strategy_v0 = strategy_v0;
  config->use_gpu_constraints = use_gpu_constraints;
  return wrap(config);
}

OptimizerConfig optimizerConfigCreateDefault() {
  return wrap(new mlir::concretelang::optimizer::Config());
}

/// ********** CompilerEngine CAPI *********************************************

CompilerEngine compilerEngineCreate() {
  auto *engine = new mlir::concretelang::CompilerEngine(
      mlir::concretelang::CompilationContext::createShared());
  return wrap(engine);
}

void compilerEngineDestroy(CompilerEngine engine) { delete unwrap(engine); }

/// Map C compilationTarget to Cpp
llvm::Expected<mlir::concretelang::CompilerEngine::Target>
targetConvertToCppFromC(CompilationTarget target) {
  switch (target) {
  case ROUND_TRIP:
    return mlir::concretelang::CompilerEngine::Target::ROUND_TRIP;
  case FHE:
    return mlir::concretelang::CompilerEngine::Target::FHE;
  case TFHE:
    return mlir::concretelang::CompilerEngine::Target::TFHE;
  case CONCRETE:
    return mlir::concretelang::CompilerEngine::Target::CONCRETE;
  case CONCRETEWITHLOOPS:
    return mlir::concretelang::CompilerEngine::Target::CONCRETEWITHLOOPS;
  case BCONCRETE:
    return mlir::concretelang::CompilerEngine::Target::BCONCRETE;
  case STD:
    return mlir::concretelang::CompilerEngine::Target::STD;
  case LLVM:
    return mlir::concretelang::CompilerEngine::Target::LLVM;
  case LLVM_IR:
    return mlir::concretelang::CompilerEngine::Target::LLVM_IR;
  case OPTIMIZED_LLVM_IR:
    return mlir::concretelang::CompilerEngine::Target::OPTIMIZED_LLVM_IR;
  case LIBRARY:
    return mlir::concretelang::CompilerEngine::Target::LIBRARY;
  }
  return mlir::concretelang::StreamStringError("invalid compilation target");
}

CompilationResult compilerEngineCompile(CompilerEngine engine,
                                        MlirStringRef module,
                                        CompilationTarget target) {
  std::string module_str(module.data, module.length);
  auto targetCppOrError = targetConvertToCppFromC(target);
  if (!targetCppOrError) { // invalid target
    llvm::errs() << llvm::toString(targetCppOrError.takeError());
    return wrap((mlir::concretelang::CompilerEngine::CompilationResult *)NULL);
  }
  auto retOrError = unwrap(engine)->compile(module_str, targetCppOrError.get());
  if (!retOrError) { // compilation error
    llvm::errs() << llvm::toString(retOrError.takeError());
    return wrap((mlir::concretelang::CompilerEngine::CompilationResult *)NULL);
  }
  return wrap(new mlir::concretelang::CompilerEngine::CompilationResult(
      std::move(retOrError.get())));
}

void compilerEngineCompileSetOptions(CompilerEngine engine,
                                     CompilationOptions options) {
  unwrap(engine)->setCompilationOptions(*unwrap(options));
}

/// ********** CompilationResult CAPI ******************************************

MlirStringRef compilationResultGetModuleString(CompilationResult result) {
  // print the module into a string
  std::string moduleString;
  llvm::raw_string_ostream os(moduleString);
  unwrap(result)->mlirModuleRef->get().print(os);
  // allocate buffer and copy module string
  char *buffer = new char[moduleString.length() + 1];
  strcpy(buffer, moduleString.c_str());
  return mlirStringRefCreate(buffer, moduleString.length());
}

void compilationResultDestroyModuleString(MlirStringRef str) {
  delete str.data;
}

void compilationResultDestroy(CompilationResult result) {
  delete unwrap(result);
}

/// ********** Library CAPI ****************************************************

Library libraryCreate(MlirStringRef outputDirPath,
                      MlirStringRef runtimeLibraryPath, bool cleanUp) {
  std::string outputDirPathStr(outputDirPath.data, outputDirPath.length);
  std::string runtimeLibraryPathStr(runtimeLibraryPath.data,
                                    runtimeLibraryPath.length);
  return wrap(new mlir::concretelang::CompilerEngine::Library(
      outputDirPathStr, runtimeLibraryPathStr, cleanUp));
}

void libraryDestroy(Library lib) { delete unwrap(lib); }

/// ********** LibraryCompilationResult CAPI ***********************************

void libraryCompilationResultDestroy(LibraryCompilationResult result) {
  delete unwrap(result);
}

/// ********** LibrarySupport CAPI *********************************************

LibrarySupport
librarySupportCreate(MlirStringRef outputDirPath,
                     MlirStringRef runtimeLibraryPath, bool generateSharedLib,
                     bool generateStaticLib, bool generateClientParameters,
                     bool generateCompilationFeedback, bool generateCppHeader) {
  std::string outputDirPathStr(outputDirPath.data, outputDirPath.length);
  std::string runtimeLibraryPathStr(runtimeLibraryPath.data,
                                    runtimeLibraryPath.length);
  return wrap(new mlir::concretelang::LibrarySupport(
      outputDirPathStr, runtimeLibraryPathStr, generateSharedLib,
      generateStaticLib, generateClientParameters, generateCompilationFeedback,
      generateCppHeader));
}

LibraryCompilationResult librarySupportCompile(LibrarySupport support,
                                               MlirStringRef module,
                                               CompilationOptions options) {
  std::string moduleStr(module.data, module.length);
  auto retOrError = unwrap(support)->compile(moduleStr, *unwrap(options));
  if (!retOrError) {
    llvm::errs() << llvm::toString(retOrError.takeError());
    return wrap((mlir::concretelang::LibraryCompilationResult *)NULL);
  }
  return wrap(new mlir::concretelang::LibraryCompilationResult(
      *retOrError.get().release()));
}

ServerLambda librarySupportLoadServerLambda(LibrarySupport support,
                                            LibraryCompilationResult result) {
  auto serverLambdaOrError = unwrap(support)->loadServerLambda(*unwrap(result));
  if (!serverLambdaOrError) {
    llvm::errs() << llvm::toString(serverLambdaOrError.takeError());
    return wrap((mlir::concretelang::serverlib::ServerLambda *)NULL);
  }
  return wrap(new mlir::concretelang::serverlib::ServerLambda(
      serverLambdaOrError.get()));
}

ClientParameters
librarySupportLoadClientParameters(LibrarySupport support,
                                   LibraryCompilationResult result) {
  auto paramsOrError = unwrap(support)->loadClientParameters(*unwrap(result));
  if (!paramsOrError) {
    llvm::errs() << llvm::toString(paramsOrError.takeError());
    return wrap((mlir::concretelang::clientlib::ClientParameters *)NULL);
  }
  return wrap(
      new mlir::concretelang::clientlib::ClientParameters(paramsOrError.get()));
}

PublicResult librarySupportServerCall(LibrarySupport support,
                                      ServerLambda server_lambda,
                                      PublicArguments args,
                                      EvaluationKeys evalKeys) {
  auto resultOrError = unwrap(support)->serverCall(
      *unwrap(server_lambda), *unwrap(args), *unwrap(evalKeys));
  if (!resultOrError) {
    llvm::errs() << llvm::toString(resultOrError.takeError());
    return wrap((mlir::concretelang::clientlib::PublicResult *)NULL);
  }
  return wrap(resultOrError.get().release());
}

void librarySupportDestroy(LibrarySupport support) { delete unwrap(support); }

/// ********** ServerLamda CAPI ************************************************

void serverLambdaDestroy(ServerLambda server) { delete unwrap(server); }

/// ********** ClientParameters CAPI *******************************************

void clientParametersDestroy(ClientParameters params) { delete unwrap(params); }

/// ********** KeySet CAPI *****************************************************

KeySet keySetGenerate(ClientParameters params, uint64_t seed_msb,
                      uint64_t seed_lsb) {
  auto keySet = mlir::concretelang::clientlib::KeySet::generate(
      *unwrap(params), seed_msb, seed_lsb);
  if (keySet.has_error()) {
    llvm::errs() << keySet.error().mesg;
    return wrap((mlir::concretelang::clientlib::KeySet *)NULL);
  }
  return wrap(keySet.value().release());
}

EvaluationKeys keySetGetEvaluationKeys(KeySet keySet) {
  return wrap(new mlir::concretelang::clientlib::EvaluationKeys(
      unwrap(keySet)->evaluationKeys()));
}

void keySetDestroy(KeySet keySet) { delete unwrap(keySet); }

/// ********** KeySetCache CAPI ************************************************

KeySetCache keySetCacheCreate(MlirStringRef cachePath) {
  std::string cachePathStr(cachePath.data, cachePath.length);
  return wrap(new mlir::concretelang::clientlib::KeySetCache(cachePathStr));
}

KeySet keySetCacheLoadOrGenerateKeySet(KeySetCache cache,
                                       ClientParameters params,
                                       uint64_t seed_msb, uint64_t seed_lsb) {
  auto keySetOrError =
      unwrap(cache)->generate(*unwrap(params), seed_msb, seed_lsb);
  if (keySetOrError.has_error()) {
    llvm::errs() << keySetOrError.error().mesg;
    return wrap((mlir::concretelang::clientlib::KeySet *)NULL);
  }
  return wrap(keySetOrError.value().release());
}

void keySetCacheDestroy(KeySetCache keySetCache) { delete unwrap(keySetCache); }

/// ********** EvaluationKeys CAPI *********************************************

void evaluationKeysDestroy(EvaluationKeys evaluationKeys) {
  delete unwrap(evaluationKeys);
}

/// ********** LambdaArgument CAPI *********************************************

LambdaArgument lambdaArgumentFromScalar(uint64_t value) {
  return wrap(new mlir::concretelang::IntLambdaArgument<uint64_t>(value));
}

// LambdaArgument lambdaArgumentFromTensorU64(uint64_t *data, int64_t *dims,
//                                            size_t rank);

bool lambdaArgumentIsScalar(LambdaArgument lambdaArg) {
  return unwrap(lambdaArg)
      ->isa<mlir::concretelang::IntLambdaArgument<uint64_t>>();
}

uint64_t lambdaArgumentGetScalar(LambdaArgument lambdaArg) {
  mlir::concretelang::IntLambdaArgument<uint64_t> *arg =
      unwrap(lambdaArg)
          ->dyn_cast<mlir::concretelang::IntLambdaArgument<uint64_t>>();
  assert(arg != nullptr && "lambda argument isn't a scalar");
  return arg->getValue();
}

bool lambdaArgumentIsTensor(LambdaArgument lambdaArg) {
  return unwrap(lambdaArg)
             ->isa<mlir::concretelang::TensorLambdaArgument<
                 mlir::concretelang::IntLambdaArgument<uint8_t>>>() ||
         unwrap(lambdaArg)
             ->isa<mlir::concretelang::TensorLambdaArgument<
                 mlir::concretelang::IntLambdaArgument<uint16_t>>>() ||
         unwrap(lambdaArg)
             ->isa<mlir::concretelang::TensorLambdaArgument<
                 mlir::concretelang::IntLambdaArgument<uint32_t>>>() ||
         unwrap(lambdaArg)
             ->isa<mlir::concretelang::TensorLambdaArgument<
                 mlir::concretelang::IntLambdaArgument<uint64_t>>>();
}

// uint64_t *lambdaArgumentGetTensorData(LambdaArgument lambdaArg);
// size_t lambdaArgumentGetTensorRank(LambdaArgument lambdaArg);
// int64_t *lambdaArgumentGetTensorDims(LambdaArgument lambdaArg);

PublicArguments lambdaArgumentEncrypt(const LambdaArgument *lambdaArgs,
                                      size_t argNumber, ClientParameters params,
                                      KeySet keySet) {
  std::vector<const mlir::concretelang::LambdaArgument *> args;
  for (size_t i = 0; i < argNumber; i++)
    args.push_back(unwrap(lambdaArgs[i]));
  auto publicArgsOrError =
      mlir::concretelang::LambdaSupport<int, int>::exportArguments(
          *unwrap(params), *unwrap(keySet), args);
  if (!publicArgsOrError) {
    llvm::errs() << llvm::toString(publicArgsOrError.takeError());
    return wrap((mlir::concretelang::clientlib::PublicArguments *)NULL);
  }
  return wrap(publicArgsOrError.get().release());
}

void lambdaArgumentDestroy(LambdaArgument lambdaArg) {
  delete unwrap(lambdaArg);
}

/// ********** PublicArguments CAPI ********************************************

void publicArgumentsDestroy(PublicArguments publicArgs) {
  delete unwrap(publicArgs);
}

/// ********** PublicResult CAPI ***********************************************

LambdaArgument publicResultDecrypt(PublicResult publicResult, KeySet keySet) {
  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>>
      lambdaArgOrError = mlir::concretelang::typedResult<
          std::unique_ptr<mlir::concretelang::LambdaArgument>>(
          *unwrap(keySet), *unwrap(publicResult));
  if (!lambdaArgOrError) {
    llvm::errs() << llvm::toString(lambdaArgOrError.takeError());
    return wrap((mlir::concretelang::LambdaArgument *)NULL);
  }
  return wrap(lambdaArgOrError.get().release());
}

void publicResultDestroy(PublicResult publicResult) {
  delete unwrap(publicResult);
}
