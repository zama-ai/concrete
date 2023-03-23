// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang-c/Support/CompilerEngine.h"
#include "concretelang/CAPI/Wrappers.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/LambdaArgument.h"
#include "concretelang/Support/LambdaSupport.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SourceMgr.h"
#include <numeric>

#define C_STRUCT_CLEANER(c_struct)                                             \
  auto *cpp = unwrap(c_struct);                                                \
  if (cpp != NULL)                                                             \
    delete cpp;                                                                \
  const char *error = getErrorPtr(c_struct);                                   \
  if (error != NULL)                                                           \
    delete[] error;

/// ********** BufferRef CAPI **************************************************

BufferRef bufferRefCreate(const char *buffer, size_t length) {
  return BufferRef{buffer, length, NULL};
}

BufferRef bufferRefFromString(std::string str) {
  char *buffer = new char[str.size()];
  memcpy(buffer, str.c_str(), str.size());
  return bufferRefCreate(buffer, str.size());
}

BufferRef bufferRefFromStringError(std::string error) {
  char *buffer = new char[error.size()];
  memcpy(buffer, error.c_str(), error.size());
  return BufferRef{NULL, 0, buffer};
}

void bufferRefDestroy(BufferRef buffer) {
  if (buffer.data != NULL)
    delete[] buffer.data;
  if (buffer.error != NULL)
    delete[] buffer.error;
}

/// ********** Utilities *******************************************************

void mlirStringRefDestroy(MlirStringRef str) { delete[] str.data; }

template <typename T> BufferRef serialize(T toSerialize) {
  std::ostringstream ostream(std::ios::binary);
  auto voidOrError = unwrap(toSerialize)->serialize(ostream);
  if (voidOrError.has_error()) {
    return bufferRefFromStringError(voidOrError.error().mesg);
  }
  return bufferRefFromString(ostream.str());
}

/// ********** CompilationOptions CAPI *****************************************

CompilationOptions
compilationOptionsCreate(MlirStringRef funcName, bool autoParallelize,
                         bool batchTFHEOps, bool dataflowParallelize,
                         bool emitGPUOps, bool loopParallelize,
                         bool optimizeTFHE, OptimizerConfig optimizerConfig,
                         bool verifyDiagnostics) {
  std::string funcNameStr(funcName.data, funcName.length);
  auto options = new mlir::concretelang::CompilationOptions(funcNameStr);
  options->autoParallelize = autoParallelize;
  options->batchTFHEOps = batchTFHEOps;
  options->dataflowParallelize = dataflowParallelize;
  options->emitGPUOps = emitGPUOps;
  options->loopParallelize = loopParallelize;
  options->optimizeTFHE = optimizeTFHE;
  options->optimizerConfig = *unwrap(optimizerConfig);
  options->verifyDiagnostics = verifyDiagnostics;
  return wrap(options);
}

CompilationOptions compilationOptionsCreateDefault() {
  return wrap(new mlir::concretelang::CompilationOptions("main"));
}

void compilationOptionsDestroy(CompilationOptions options){
    C_STRUCT_CLEANER(options)}

/// ********** OptimizerConfig CAPI ********************************************

OptimizerConfig
    optimizerConfigCreate(bool display, double fallback_log_norm_woppbs,
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

void optimizerConfigDestroy(OptimizerConfig config){C_STRUCT_CLEANER(config)}

/// ********** CompilerEngine CAPI *********************************************

CompilerEngine compilerEngineCreate() {
  auto *engine = new mlir::concretelang::CompilerEngine(
      mlir::concretelang::CompilationContext::createShared());
  return wrap(engine);
}

void compilerEngineDestroy(CompilerEngine engine){C_STRUCT_CLEANER(engine)}

/// Map C compilationTarget to Cpp
llvm::Expected<mlir::concretelang::CompilerEngine::
                   Target> targetConvertToCppFromC(CompilationTarget target) {
  switch (target) {
  case ROUND_TRIP:
    return mlir::concretelang::CompilerEngine::Target::ROUND_TRIP;
  case FHE:
    return mlir::concretelang::CompilerEngine::Target::FHE;
  case TFHE:
    return mlir::concretelang::CompilerEngine::Target::TFHE;
  case CONCRETE:
    return mlir::concretelang::CompilerEngine::Target::CONCRETE;
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
    return wrap((mlir::concretelang::CompilerEngine::CompilationResult *)NULL,
                llvm::toString(targetCppOrError.takeError()));
  }
  auto retOrError = unwrap(engine)->compile(module_str, targetCppOrError.get());
  if (!retOrError) { // compilation error
    return wrap((mlir::concretelang::CompilerEngine::CompilationResult *)NULL,
                llvm::toString(retOrError.takeError()));
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
  mlirStringRefDestroy(str);
}

void compilationResultDestroy(CompilationResult result){
    C_STRUCT_CLEANER(result)}

/// ********** Library CAPI ****************************************************

Library libraryCreate(MlirStringRef outputDirPath,
                      MlirStringRef runtimeLibraryPath, bool cleanUp) {
  std::string outputDirPathStr(outputDirPath.data, outputDirPath.length);
  std::string runtimeLibraryPathStr(runtimeLibraryPath.data,
                                    runtimeLibraryPath.length);
  return wrap(new mlir::concretelang::CompilerEngine::Library(
      outputDirPathStr, runtimeLibraryPathStr, cleanUp));
}

void libraryDestroy(Library lib) { C_STRUCT_CLEANER(lib) }

/// ********** LibraryCompilationResult CAPI ***********************************

void libraryCompilationResultDestroy(LibraryCompilationResult result){
    C_STRUCT_CLEANER(result)}

/// ********** LibrarySupport CAPI *********************************************

LibrarySupport
    librarySupportCreate(MlirStringRef outputDirPath,
                         MlirStringRef runtimeLibraryPath,
                         bool generateSharedLib, bool generateStaticLib,
                         bool generateClientParameters,
                         bool generateCompilationFeedback,
                         bool generateCppHeader) {
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
    return wrap((mlir::concretelang::LibraryCompilationResult *)NULL,
                llvm::toString(retOrError.takeError()));
  }
  return wrap(new mlir::concretelang::LibraryCompilationResult(
      *retOrError.get().release()));
}

ServerLambda librarySupportLoadServerLambda(LibrarySupport support,
                                            LibraryCompilationResult result) {
  auto serverLambdaOrError = unwrap(support)->loadServerLambda(*unwrap(result));
  if (!serverLambdaOrError) {
    return wrap((mlir::concretelang::serverlib::ServerLambda *)NULL,
                llvm::toString(serverLambdaOrError.takeError()));
  }
  return wrap(new mlir::concretelang::serverlib::ServerLambda(
      serverLambdaOrError.get()));
}

ClientParameters
librarySupportLoadClientParameters(LibrarySupport support,
                                   LibraryCompilationResult result) {
  auto paramsOrError = unwrap(support)->loadClientParameters(*unwrap(result));
  if (!paramsOrError) {
    return wrap((mlir::concretelang::clientlib::ClientParameters *)NULL,
                llvm::toString(paramsOrError.takeError()));
  }
  return wrap(
      new mlir::concretelang::clientlib::ClientParameters(paramsOrError.get()));
}

CompilationFeedback
librarySupportLoadCompilationFeedback(LibrarySupport support,
                                      LibraryCompilationResult result) {
  auto feedbackOrError =
      unwrap(support)->loadCompilationFeedback(*unwrap(result));
  if (!feedbackOrError) {
    return wrap((mlir::concretelang::CompilationFeedback *)NULL,
                llvm::toString(feedbackOrError.takeError()));
  }
  return wrap(
      new mlir::concretelang::CompilationFeedback(feedbackOrError.get()));
}

PublicResult librarySupportServerCall(LibrarySupport support,
                                      ServerLambda server_lambda,
                                      PublicArguments args,
                                      EvaluationKeys evalKeys) {
  auto resultOrError = unwrap(support)->serverCall(
      *unwrap(server_lambda), *unwrap(args), *unwrap(evalKeys));
  if (!resultOrError) {
    return wrap((mlir::concretelang::clientlib::PublicResult *)NULL,
                llvm::toString(resultOrError.takeError()));
  }
  return wrap(resultOrError.get().release());
}

MlirStringRef librarySupportGetSharedLibPath(LibrarySupport support) {
  auto path = unwrap(support)->getSharedLibPath();
  // allocate buffer and copy module string
  char *buffer = new char[path.length() + 1];
  strcpy(buffer, path.c_str());
  return mlirStringRefCreate(buffer, path.length());
}

MlirStringRef librarySupportGetClientParametersPath(LibrarySupport support) {
  auto path = unwrap(support)->getClientParametersPath();
  // allocate buffer and copy module string
  char *buffer = new char[path.length() + 1];
  strcpy(buffer, path.c_str());
  return mlirStringRefCreate(buffer, path.length());
}

void librarySupportDestroy(LibrarySupport support) { C_STRUCT_CLEANER(support) }

/// ********** ServerLamda CAPI ************************************************

void serverLambdaDestroy(ServerLambda server){C_STRUCT_CLEANER(server)}

/// ********** ClientParameters CAPI *******************************************

BufferRef clientParametersSerialize(ClientParameters params) {
  llvm::json::Value value(*unwrap(params));
  std::string jsonParams;
  llvm::raw_string_ostream ostream(jsonParams);
  ostream << value;
  char *buffer = new char[jsonParams.size() + 1];
  strcpy(buffer, jsonParams.c_str());
  return bufferRefCreate(buffer, jsonParams.size());
}

ClientParameters clientParametersUnserialize(BufferRef buffer) {
  std::string json(buffer.data, buffer.length);
  auto paramsOrError =
      llvm::json::parse<mlir::concretelang::ClientParameters>(json);
  if (!paramsOrError) {
    return wrap((mlir::concretelang::ClientParameters *)NULL,
                llvm::toString(paramsOrError.takeError()));
  }
  return wrap(new mlir::concretelang::ClientParameters(paramsOrError.get()));
}

ClientParameters clientParametersCopy(ClientParameters params) {
  return wrap(new mlir::concretelang::ClientParameters(*unwrap(params)));
}

void clientParametersDestroy(ClientParameters params){C_STRUCT_CLEANER(params)}

size_t clientParametersOutputsSize(ClientParameters params) {
  return unwrap(params)->outputs.size();
}

size_t clientParametersInputsSize(ClientParameters params) {
  return unwrap(params)->inputs.size();
}

CircuitGate clientParametersOutputCircuitGate(ClientParameters params,
                                              size_t index) {
  auto &cppGate = unwrap(params)->outputs[index];
  auto *cppGateCopy = new mlir::concretelang::clientlib::CircuitGate(cppGate);
  return wrap(cppGateCopy);
}

CircuitGate clientParametersInputCircuitGate(ClientParameters params,
                                             size_t index) {
  auto &cppGate = unwrap(params)->inputs[index];
  auto *cppGateCopy = new mlir::concretelang::clientlib::CircuitGate(cppGate);
  return wrap(cppGateCopy);
}

EncryptionGate circuitGateEncryptionGate(CircuitGate circuit_gate) {
  auto &maybe_gate = unwrap(circuit_gate)->encryption;

  if (maybe_gate) {
    auto *copy = new mlir::concretelang::clientlib::EncryptionGate(*maybe_gate);
    return wrap(copy);
  }
  return (static_cast<EncryptionGate (*)(
              mlir::concretelang::clientlib::EncryptionGate *)>(wrap))(nullptr);
}

double encryptionGateVariance(EncryptionGate encryption_gate) {
  return unwrap(encryption_gate)->variance;
}

Encoding encryptionGateEncoding(EncryptionGate encryption_gate) {
  auto &cppEncoding = unwrap(encryption_gate)->encoding;
  auto *copy = new mlir::concretelang::clientlib::Encoding(cppEncoding);
  return wrap(copy);
}

uint64_t encodingPrecision(Encoding encoding) {
  return unwrap(encoding)->precision;
}

void circuitGateDestroy(CircuitGate gate) { C_STRUCT_CLEANER(gate) }
void encryptionGateDestroy(EncryptionGate gate) { C_STRUCT_CLEANER(gate) }
void encodingDestroy(Encoding encoding){C_STRUCT_CLEANER(encoding)}

/// ********** KeySet CAPI *****************************************************

KeySet keySetGenerate(ClientParameters params, uint64_t seed_msb,
                      uint64_t seed_lsb) {

  __uint128_t seed = seed_msb;
  seed <<= 64;
  seed += seed_lsb;

  auto csprng = concretelang::clientlib::ConcreteCSPRNG(seed);

  auto keySet = mlir::concretelang::clientlib::KeySet::generate(
      *unwrap(params), std::move(csprng));
  if (keySet.has_error()) {
    return wrap((mlir::concretelang::clientlib::KeySet *)NULL,
                keySet.error().mesg);
  }
  return wrap(keySet.value().release());
}

EvaluationKeys keySetGetEvaluationKeys(KeySet keySet) {
  return wrap(new mlir::concretelang::clientlib::EvaluationKeys(
      unwrap(keySet)->evaluationKeys()));
}

void keySetDestroy(KeySet keySet){C_STRUCT_CLEANER(keySet)}

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
    return wrap((mlir::concretelang::clientlib::KeySet *)NULL,
                keySetOrError.error().mesg);
  }
  return wrap(keySetOrError.value().release());
}

void keySetCacheDestroy(KeySetCache keySetCache){C_STRUCT_CLEANER(keySetCache)}

/// ********** EvaluationKeys CAPI *********************************************

BufferRef evaluationKeysSerialize(EvaluationKeys keys) {
  std::ostringstream ostream(std::ios::binary);
  concretelang::clientlib::operator<<(ostream, *unwrap(keys));
  if (ostream.fail()) {
    return bufferRefFromStringError(
        "output stream failure during evaluation keys serialization");
  }
  return bufferRefFromString(ostream.str());
}

EvaluationKeys evaluationKeysUnserialize(BufferRef buffer) {
  std::stringstream istream(std::string(buffer.data, buffer.length));
  concretelang::clientlib::EvaluationKeys evaluationKeys =
      concretelang::clientlib::readEvaluationKeys(istream);
  if (istream.fail()) {
    return wrap((concretelang::clientlib::EvaluationKeys *)NULL,
                "input stream failure during evaluation keys unserialization");
  }
  return wrap(new concretelang::clientlib::EvaluationKeys(evaluationKeys));
}

void evaluationKeysDestroy(EvaluationKeys evaluationKeys) {
  C_STRUCT_CLEANER(evaluationKeys);
}

/// ********** LambdaArgument CAPI *********************************************

LambdaArgument lambdaArgumentFromScalar(uint64_t value) {
  return wrap(new mlir::concretelang::IntLambdaArgument<uint64_t>(value));
}

int64_t getSizeFromRankAndDims(size_t rank, const int64_t *dims) {
  if (rank == 0) // not a tensor
    return 1;
  auto size = dims[0];
  for (size_t i = 1; i < rank; i++)
    size *= dims[i];
  return size;
}

LambdaArgument lambdaArgumentFromTensorU8(const uint8_t *data,
                                          const int64_t *dims, size_t rank) {

  std::vector<uint8_t> data_vector(data,
                                   data + getSizeFromRankAndDims(rank, dims));
  std::vector<int64_t> dims_vector(dims, dims + rank);
  return wrap(new mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint8_t>>(data_vector,
                                                              dims_vector));
}

LambdaArgument lambdaArgumentFromTensorU16(const uint16_t *data,
                                           const int64_t *dims, size_t rank) {

  std::vector<uint16_t> data_vector(data,
                                    data + getSizeFromRankAndDims(rank, dims));
  std::vector<int64_t> dims_vector(dims, dims + rank);
  return wrap(new mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint16_t>>(data_vector,
                                                               dims_vector));
}

LambdaArgument lambdaArgumentFromTensorU32(const uint32_t *data,
                                           const int64_t *dims, size_t rank) {

  std::vector<uint32_t> data_vector(data,
                                    data + getSizeFromRankAndDims(rank, dims));
  std::vector<int64_t> dims_vector(dims, dims + rank);
  return wrap(new mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint32_t>>(data_vector,
                                                               dims_vector));
}

LambdaArgument lambdaArgumentFromTensorU64(const uint64_t *data,
                                           const int64_t *dims, size_t rank) {

  std::vector<uint64_t> data_vector(data,
                                    data + getSizeFromRankAndDims(rank, dims));
  std::vector<int64_t> dims_vector(dims, dims + rank);
  return wrap(new mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint64_t>>(data_vector,
                                                               dims_vector));
}

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

template <typename T>
bool copyTensorDataToBuffer(
    mlir::concretelang::TensorLambdaArgument<
        mlir::concretelang::IntLambdaArgument<T>> *tensor,
    uint64_t *buffer) {
  auto *data = tensor->getValue();
  auto sizeOrError = tensor->getNumElements();
  if (!sizeOrError) {
    llvm::errs() << llvm::toString(sizeOrError.takeError());
    return false;
  }
  auto size = sizeOrError.get();
  for (size_t i = 0; i < size; i++)
    buffer[i] = data[i];
  return true;
}

bool lambdaArgumentGetTensorData(LambdaArgument lambdaArg, uint64_t *buffer) {
  auto arg = unwrap(lambdaArg);
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint8_t>>>()) {
    return copyTensorDataToBuffer(tensor, buffer);
  }
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint16_t>>>()) {
    return copyTensorDataToBuffer(tensor, buffer);
  }
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint32_t>>>()) {
    return copyTensorDataToBuffer(tensor, buffer);
  }
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint64_t>>>()) {
    return copyTensorDataToBuffer(tensor, buffer);
  }
  return false;
}

size_t lambdaArgumentGetTensorRank(LambdaArgument lambdaArg) {
  auto arg = unwrap(lambdaArg);
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint8_t>>>()) {
    return tensor->getDimensions().size();
  }
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint16_t>>>()) {
    return tensor->getDimensions().size();
  }
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint32_t>>>()) {
    return tensor->getDimensions().size();
  }
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint64_t>>>()) {
    return tensor->getDimensions().size();
  }
  return 0;
}

int64_t lambdaArgumentGetTensorDataSize(LambdaArgument lambdaArg) {
  auto arg = unwrap(lambdaArg);
  std::vector<int64_t> dims;
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint8_t>>>()) {
    dims = tensor->getDimensions();
  } else if (auto tensor =
                 arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                     mlir::concretelang::IntLambdaArgument<uint16_t>>>()) {
    dims = tensor->getDimensions();
  } else if (auto tensor =
                 arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                     mlir::concretelang::IntLambdaArgument<uint32_t>>>()) {
    dims = tensor->getDimensions();
  } else if (auto tensor =
                 arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                     mlir::concretelang::IntLambdaArgument<uint64_t>>>()) {
    dims = tensor->getDimensions();
  } else {
    return 0;
  }
  return std::accumulate(std::begin(dims), std::end(dims), 1,
                         std::multiplies<int64_t>());
}

bool lambdaArgumentGetTensorDims(LambdaArgument lambdaArg, int64_t *buffer) {
  auto arg = unwrap(lambdaArg);
  std::vector<int64_t> dims;
  if (auto tensor = arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                        mlir::concretelang::IntLambdaArgument<uint8_t>>>()) {
    dims = tensor->getDimensions();
  } else if (auto tensor =
                 arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                     mlir::concretelang::IntLambdaArgument<uint16_t>>>()) {
    dims = tensor->getDimensions();
  } else if (auto tensor =
                 arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                     mlir::concretelang::IntLambdaArgument<uint32_t>>>()) {
    dims = tensor->getDimensions();
  } else if (auto tensor =
                 arg->dyn_cast<mlir::concretelang::TensorLambdaArgument<
                     mlir::concretelang::IntLambdaArgument<uint64_t>>>()) {
    dims = tensor->getDimensions();
  } else {
    return false;
  }
  memcpy(buffer, dims.data(), sizeof(int64_t) * dims.size());
  return true;
}

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
    return wrap((mlir::concretelang::clientlib::PublicArguments *)NULL,
                llvm::toString(publicArgsOrError.takeError()));
  }
  return wrap(publicArgsOrError.get().release());
}

void lambdaArgumentDestroy(LambdaArgument lambdaArg){
    C_STRUCT_CLEANER(lambdaArg)}

/// ********** PublicArguments CAPI ********************************************

BufferRef publicArgumentsSerialize(PublicArguments args) {
  return serialize(args);
}

PublicArguments publicArgumentsUnserialize(BufferRef buffer,
                                           ClientParameters params) {
  std::stringstream istream(std::string(buffer.data, buffer.length));
  auto argsOrError = concretelang::clientlib::PublicArguments::unserialize(
      *unwrap(params), istream);
  if (!argsOrError) {
    return wrap((concretelang::clientlib::PublicArguments *)NULL,
                argsOrError.error().mesg);
  }
  return wrap(argsOrError.value().release());
}

void publicArgumentsDestroy(PublicArguments publicArgs){
    C_STRUCT_CLEANER(publicArgs)}

/// ********** PublicResult CAPI ***********************************************

LambdaArgument publicResultDecrypt(PublicResult publicResult, KeySet keySet) {
  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>>
      lambdaArgOrError = mlir::concretelang::typedResult<
          std::unique_ptr<mlir::concretelang::LambdaArgument>>(
          *unwrap(keySet), *unwrap(publicResult));
  if (!lambdaArgOrError) {
    return wrap((mlir::concretelang::LambdaArgument *)NULL,
                llvm::toString(lambdaArgOrError.takeError()));
  }
  return wrap(lambdaArgOrError.get().release());
}

BufferRef publicResultSerialize(PublicResult result) {
  return serialize(result);
}

PublicResult publicResultUnserialize(BufferRef buffer,
                                     ClientParameters params) {
  std::stringstream istream(std::string(buffer.data, buffer.length));
  auto resultOrError = concretelang::clientlib::PublicResult::unserialize(
      *unwrap(params), istream);
  if (!resultOrError) {
    return wrap((concretelang::clientlib::PublicResult *)NULL,
                resultOrError.error().mesg);
  }
  return wrap(resultOrError.value().release());
}

void publicResultDestroy(PublicResult publicResult) {
  C_STRUCT_CLEANER(publicResult)
}

/// ********** CompilationFeedback CAPI ****************************************

double compilationFeedbackGetComplexity(CompilationFeedback feedback) {
  return unwrap(feedback)->complexity;
}

double compilationFeedbackGetPError(CompilationFeedback feedback) {
  return unwrap(feedback)->pError;
}

double compilationFeedbackGetGlobalPError(CompilationFeedback feedback) {
  return unwrap(feedback)->globalPError;
}

uint64_t
compilationFeedbackGetTotalSecretKeysSize(CompilationFeedback feedback) {
  return unwrap(feedback)->totalSecretKeysSize;
}

uint64_t
compilationFeedbackGetTotalBootstrapKeysSize(CompilationFeedback feedback) {
  return unwrap(feedback)->totalBootstrapKeysSize;
}

uint64_t
compilationFeedbackGetTotalKeyswitchKeysSize(CompilationFeedback feedback) {
  return unwrap(feedback)->totalKeyswitchKeysSize;
}

uint64_t compilationFeedbackGetTotalInputsSize(CompilationFeedback feedback) {
  return unwrap(feedback)->totalInputsSize;
}

uint64_t compilationFeedbackGetTotalOutputsSize(CompilationFeedback feedback) {
  return unwrap(feedback)->totalOutputsSize;
}

void compilationFeedbackDestroy(CompilationFeedback feedback) {
  C_STRUCT_CLEANER(feedback)
}
