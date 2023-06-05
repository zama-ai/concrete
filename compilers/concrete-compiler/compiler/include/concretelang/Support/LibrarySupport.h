// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_LIBRARY_SUPPORT
#define CONCRETELANG_SUPPORT_LIBRARY_SUPPORT

#include "concrete-protocol.pb.h"
#include "concretelang/Common/Protobuf.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <concretelang/Common/Protobuf.h>
#include <concretelang/ServerLib/ServerLambda.h>
#include <concretelang/Support/CompilerEngine.h>
#include <concretelang/Support/Jit.h>
#include <concretelang/Support/LambdaSupport.h>

namespace mlir {
namespace concretelang {

namespace clientlib = ::concretelang::clientlib;
namespace serverlib = ::concretelang::serverlib;

/// LibraryCompilationResult is the result of a compilation to a library.
struct LibraryCompilationResult {
  /// The output directory path where the compilation artifacts have been
  /// generated.
  std::string outputDirPath;
  std::string funcName;
};

class LibrarySupport
    : public LambdaSupport<serverlib::ServerLambda, LibraryCompilationResult> {

public:
  LibrarySupport(std::string outputPath, std::string runtimeLibraryPath = "",
                 bool generateSharedLib = true, bool generateStaticLib = true,
                 bool generateClientParameters = true,
                 bool generateCompilationFeedback = true,
                 bool generateCppHeader = true)
      : outputPath(outputPath), runtimeLibraryPath(runtimeLibraryPath),
        generateSharedLib(generateSharedLib),
        generateStaticLib(generateStaticLib),
        generateClientParameters(generateClientParameters),
        generateCompilationFeedback(generateCompilationFeedback),
        generateCppHeader(generateCppHeader) {}

  llvm::Expected<std::unique_ptr<LibraryCompilationResult>>
  compile(llvm::SourceMgr &program, CompilationOptions options) override {
    // Setup the compiler engine
    auto context = CompilationContext::createShared();
    concretelang::CompilerEngine engine(context);
    engine.setCompilationOptions(options);

    // Compile to a library
    auto library = engine.compile(
        program, outputPath, runtimeLibraryPath, generateSharedLib,
        generateStaticLib, generateClientParameters,
        generateCompilationFeedback, generateCppHeader);
    if (auto err = library.takeError()) {
      return std::move(err);
    }

    if (!options.mainFuncName.has_value()) {
      return StreamStringError("Need to have a funcname to compile library");
    }

    auto result = std::make_unique<LibraryCompilationResult>();
    result->outputDirPath = outputPath;
    result->funcName = *options.mainFuncName;
    return std::move(result);
  }
  using LambdaSupport::compile;

  /// Load the server lambda from the compilation result.
  llvm::Expected<serverlib::ServerLambda>
  loadServerLambda(LibraryCompilationResult &result) override {
    auto lambda =
        serverlib::ServerLambda::load(result.funcName, result.outputDirPath);
    if (lambda.has_error()) {
      return StreamStringError(lambda.error().mesg);
    }
    return lambda.value();
  }

  /// Load the client parameters from the compilation result.
  llvm::Expected<clientlib::ClientParameters>
  loadClientParameters(LibraryCompilationResult &result) override {
    auto path =
        CompilerEngine::Library::getProgramInfoPath(result.outputDirPath);
    auto maybeProgramInfo = ::concretelang::common::JSONFileToMessage<
        concreteprotocol::ProgramInfo>(path);
    if (maybeProgramInfo.has_error()) {
      return StreamStringError(maybeProgramInfo.error().mesg);
    }
    concreteprotocol::ProgramInfo programInfo = maybeProgramInfo.value();
    if (programInfo.circuits_size() == 0) {
      return StreamStringError(
          "ClientLambda: Provided program info contains no circuits.");
    }
    if (programInfo.circuits_size() > 1) {
      return StreamStringError("ClientLambda: Provided program info contains "
                               "more than one circuit.");
    }
    auto param = ClientParameters::fromProgramInfo(programInfo);
    return param;
  }

  std::string getFuncName() {
    auto path = CompilerEngine::Library::getProgramInfoPath(outputPath);
    auto maybeProgramInfo = ::concretelang::common::JSONFileToMessage<
        concreteprotocol::ProgramInfo>(path);
    if (maybeProgramInfo.has_error()) {
      return "";
    }
    auto programInfo = maybeProgramInfo.value();
    auto params = ClientParameters::fromProgramInfo(programInfo);
    return params.functionName;
  }

  /// Load the the compilation result if circuit already compiled
  llvm::Expected<std::unique_ptr<LibraryCompilationResult>>
  loadCompilationResult() {
    auto funcName = getFuncName();
    if (funcName.empty()) {
      return StreamStringError("couldn't find function name");
    }
    auto result = std::make_unique<LibraryCompilationResult>();
    result->outputDirPath = outputPath;
    result->funcName = funcName;
    return std::move(result);
  }

  llvm::Expected<CompilationFeedback>
  loadCompilationFeedback(LibraryCompilationResult &result) override {
    auto path = CompilerEngine::Library::getCompilationFeedbackPath(
        result.outputDirPath);
    auto feedback = CompilationFeedback::load(path);
    if (feedback.has_error()) {
      return StreamStringError(feedback.error().mesg);
    }
    return feedback.value();
  }

  /// Call the lambda with the public arguments.
  llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
  serverCall(serverlib::ServerLambda lambda, clientlib::PublicArguments &args,
             clientlib::EvaluationKeys &evaluationKeys) override {
    return lambda.call(args, evaluationKeys);
  }

  /// Get path to shared library
  std::string getSharedLibPath() {
    return CompilerEngine::Library::getSharedLibraryPath(outputPath);
  }

  /// Get path to client parameters file
  std::string getProgramInfoPath() {
    return CompilerEngine::Library::getProgramInfoPath(outputPath);
  }

private:
  std::string outputPath;
  std::string runtimeLibraryPath;
  /// Flags to select generated artifacts
  bool generateSharedLib;
  bool generateStaticLib;
  bool generateClientParameters;
  bool generateCompilationFeedback;
  bool generateCppHeader;
};

} // namespace concretelang
} // namespace mlir

#endif
