// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TESTLIB_TESTCIRCUIT_H
#define CONCRETELANG_TESTLIB_TESTCIRCUIT_H

#include "boost/outcome.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Values.h"
#include "concretelang/ServerLib/ServerLib.h"
#include "concretelang/Support/CompilerEngine.h"
#include "tests_tools/keySetCache.h"
#include "llvm/Support/Path.h"
#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <unistd.h>

using concretelang::clientlib::ClientCircuit;
using concretelang::clientlib::ClientProgram;
using concretelang::error::Result;
using concretelang::keysets::Keyset;
using concretelang::serverlib::ServerCircuit;
using concretelang::serverlib::ServerProgram;
using concretelang::values::TransportValue;
using concretelang::values::Value;

namespace concretelang {
namespace testlib {

class TestProgram {

public:
  TestProgram(mlir::concretelang::CompilationOptions options)
      : artifactDirectory(createTempFolderIn(getSystemTempFolderPath())),
        compiler(mlir::concretelang::CompilationContext::createShared()),
        encryptionCsprng(std::make_shared<csprng::EncryptionCSPRNG>(0)) {
    compiler.setCompilationOptions(options);
  }

  TestProgram(TestProgram &&tc)
      : artifactDirectory(tc.artifactDirectory), compiler(tc.compiler),
        library(tc.library), keyset(tc.keyset),
        encryptionCsprng(tc.encryptionCsprng) {
    tc.artifactDirectory = "";
  };

  TestProgram(TestProgram &tc) = delete;

  ~TestProgram() {
    auto d = getArtifactDirectory();
    if (d.empty())
      return;
    deleteFolder(d);
  };

  Result<void> compile(std::string mlirProgram) {
    auto compilationResult = compiler.compile({mlirProgram}, artifactDirectory);
    if (!compilationResult) {
      return StringError(llvm::toString(compilationResult.takeError()));
    }
    library = compilationResult.get();
    return outcome::success();
  }

  Result<void> generateKeyset(__uint128_t secretSeed = 0,
                              __uint128_t encryptionSeed = 0,
                              bool tryCache = true) {
    if (isSimulation()) {
      keyset = Keyset{};
      return outcome::success();
    }
    OUTCOME_TRY(auto lib, getLibrary());
    OUTCOME_TRY(auto programInfo, lib.getProgramInfo());
    if (tryCache) {
      OUTCOME_TRY(keyset, getTestKeySetCachePtr()->getKeyset(
                              programInfo.asReader().getKeyset(), secretSeed,
                              encryptionSeed));
    } else {
      auto encryptionCsprng = csprng::EncryptionCSPRNG(encryptionSeed);
      auto secretCsprng = csprng::SecretCSPRNG(secretSeed);
      Message<concreteprotocol::KeysetInfo> keysetInfo =
          programInfo.asReader().getKeyset();
      keyset = Keyset(keysetInfo, secretCsprng, encryptionCsprng);
    }
    return outcome::success();
  }

  Result<std::vector<Value>> call(std::vector<Value> inputs,
                                  std::string name = "main") {
    // preprocess arguments
    auto preparedArgs = std::vector<TransportValue>();
    OUTCOME_TRY(auto clientCircuit, getClientCircuit(name));
    for (size_t i = 0; i < inputs.size(); i++) {
      OUTCOME_TRY(auto preparedInput, clientCircuit.prepareInput(inputs[i], i));
      preparedArgs.push_back(preparedInput);
    }
    // Call server
    OUTCOME_TRY(auto returns, callServer(preparedArgs, name));
    // postprocess arguments
    std::vector<Value> processedOutputs(returns.size());
    for (size_t i = 0; i < processedOutputs.size(); i++) {
      OUTCOME_TRY(processedOutputs[i],
                  clientCircuit.processOutput(returns[i], i));
    }
    return processedOutputs;
  }

  Result<std::vector<Value>> simulate(std::vector<Value> inputs,
                                      std::string name = "main") {
    // preprocess arguments
    auto preparedArgs = std::vector<TransportValue>();
    OUTCOME_TRY(auto clientCircuit, getClientCircuit(name));
    for (size_t i = 0; i < inputs.size(); i++) {
      OUTCOME_TRY(auto preparedInput,
                  clientCircuit.simulatePrepareInput(inputs[i], i));
      preparedArgs.push_back(preparedInput);
    }
    // Call server
    OUTCOME_TRY(auto returns, callServer(preparedArgs, name));
    // postprocess arguments
    std::vector<Value> processedOutputs(returns.size());
    for (size_t i = 0; i < processedOutputs.size(); i++) {
      OUTCOME_TRY(processedOutputs[i],
                  clientCircuit.simulateProcessOutput(returns[i], i));
    }
    return processedOutputs;
  }

  Result<std::vector<Value>> compose_n_times(std::vector<Value> inputs,
                                             size_t n,
                                             std::string name = "main") {
    // preprocess arguments
    auto preparedArgs = std::vector<TransportValue>();
    OUTCOME_TRY(auto clientCircuit, getClientCircuit(name));
    for (size_t i = 0; i < inputs.size(); i++) {
      OUTCOME_TRY(auto preparedInput, clientCircuit.prepareInput(inputs[i], i));
      preparedArgs.push_back(preparedInput);
    }
    // Call server multiple times in a row
    for (size_t i = 0; i < n; i++) {
      OUTCOME_TRY(preparedArgs, callServer(preparedArgs, name));
    }
    // postprocess arguments
    std::vector<Value> processedOutputs(preparedArgs.size());
    for (size_t i = 0; i < processedOutputs.size(); i++) {
      OUTCOME_TRY(processedOutputs[i],
                  clientCircuit.processOutput(preparedArgs[i], i));
    }
    return processedOutputs;
  }

  Result<std::vector<TransportValue>>
  callServer(std::vector<TransportValue> inputs, std::string name = "main") {
    std::vector<TransportValue> returns;
    OUTCOME_TRY(auto serverCircuit, getServerCircuit(name));
    if (compiler.getCompilationOptions().simulate) {
      OUTCOME_TRY(returns, serverCircuit.simulate(inputs));
    } else {
      OUTCOME_TRY(returns, serverCircuit.call(keyset->server, inputs));
    }
    return returns;
  }

  Result<ClientCircuit> getClientCircuit(std::string name = "main") {
    OUTCOME_TRY(auto lib, getLibrary());
    Keyset ks{};
    OUTCOME_TRY(auto programInfo, lib.getProgramInfo());

    if (!isSimulation()) {
      OUTCOME_TRY(ks, getKeyset());
      OUTCOME_TRY(auto clientProgram,
                  ClientProgram::createEncrypted(programInfo, ks.client,
                                                 encryptionCsprng));
      OUTCOME_TRY(auto clientCircuit, clientProgram.getClientCircuit(name));
      return clientCircuit;
    } else {
      OUTCOME_TRY(auto clientProgram, ClientProgram::createSimulated(
                                          programInfo, encryptionCsprng));
      OUTCOME_TRY(auto clientCircuit, clientProgram.getClientCircuit(name));
      return clientCircuit;
    }
  }

  Result<ServerCircuit> getServerCircuit(std::string name = "main") {
    OUTCOME_TRY(auto lib, getLibrary());
    OUTCOME_TRY(auto programInfo, lib.getProgramInfo());
    OUTCOME_TRY(auto serverProgram,
                ServerProgram::load(programInfo, lib.getSharedLibraryPath(),
                                    isSimulation()));
    OUTCOME_TRY(auto serverCircuit, serverProgram.getServerCircuit(name));
    return serverCircuit;
  }

  bool isSimulation() { return compiler.getCompilationOptions().simulate; }

private:
  std::string getArtifactDirectory() { return artifactDirectory; }

  Result<mlir::concretelang::CompilerEngine::Library> getLibrary() {
    if (!library.has_value()) {
      return StringError("TestProgram: compilation has not been done\n");
    }
    return *library;
  }

  Result<Keyset> getKeyset() {
    if (!keyset.has_value()) {
      return StringError("TestProgram: keyset has not been generated\n");
    }
    return *keyset;
  }

  std::string artifactDirectory;
  mlir::concretelang::CompilerEngine compiler;
  std::optional<mlir::concretelang::CompilerEngine::Library> library;
  std::optional<Keyset> keyset;
  std::shared_ptr<csprng::EncryptionCSPRNG> encryptionCsprng;

private:
  std::string getSystemTempFolderPath() {
    llvm::SmallString<0> tempPath;
    llvm::sys::path::system_temp_directory(true, tempPath);
    return std::string(tempPath);
  }

  void deleteFolder(const std::string &folder) {
    auto ec = std::error_code();
    llvm::errs() << "TestProgram: delete artifact directory(" << folder
                 << ")\n";
    if (!std::filesystem::remove_all(folder, ec)) {
      llvm::errs() << "TestProgram: fail to delete directory(" << folder
                   << "), error(" << ec.message() << ")\n";
    }
  }

  std::string createTempFolderIn(const std::string &rootFolder) {
    std::srand(std::time(nullptr));
    auto new_path = [=]() {
      llvm::SmallString<0> outputPath;
      llvm::sys::path::append(outputPath, rootFolder);
      auto pid = getpid();
      std::string uid = std::to_string(pid);
      uid.append("-");
      uid.append(std::to_string(std::rand()));
      llvm::sys::path::append(outputPath, uid);
      return std::string(outputPath);
    };

    auto pathString = new_path();
    auto ec = std::error_code();
    llvm::errs() << "TestProgram: create temporary directory(" << pathString
                 << ")\n";
    if (!std::filesystem::create_directory(pathString, ec)) {
      llvm::errs() << "TestProgram: fail to create temporary directory("
                   << pathString << "), ";
      if (ec) {
        llvm::errs() << "already exists";
      } else {
        llvm::errs() << "error(" << ec.message() << ")";
      }
      assert(false);
    } else {
      llvm::errs() << "TestProgram: directory(" << pathString
                   << ") successfully created\n";
      return pathString;
    }
  }
};

const std::string FUNCNAME = "main";

std::vector<uint8_t> values_3bits() { return {0, 1, 2, 5, 7}; }
std::vector<uint8_t> values_6bits() { return {0, 1, 2, 13, 22, 59, 62, 63}; }
std::vector<uint8_t> values_7bits() { return {0, 1, 2, 63, 64, 65, 125, 126}; }

} // namespace testlib
} // namespace concretelang

#endif
