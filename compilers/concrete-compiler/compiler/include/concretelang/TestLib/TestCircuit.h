// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
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

using concretelang::clientlib::ClientCircuit;
using concretelang::clientlib::ClientProgram;
using concretelang::csprng::ConcreteCSPRNG;
using concretelang::error::Result;
using concretelang::keysets::Keyset;
using concretelang::serverlib::ServerCircuit;
using concretelang::serverlib::ServerProgram;
using concretelang::values::TransportValue;
using concretelang::values::Value;

namespace concretelang {
namespace testlib {

class TestCircuit {

public:
  static Result<TestCircuit>
  create(Keyset keyset, Message<concreteprotocol::ProgramInfo> programInfo,
         std::string sharedLibPath, uint64_t seedMsb, uint64_t seedLsb,
         bool useSimulation = false) {
    OUTCOME_TRY(auto serverProgram,
                ServerProgram::load(programInfo, sharedLibPath, useSimulation));
    OUTCOME_TRY(auto serverCircuit,
                serverProgram.getServerCircuit(
                    programInfo.asReader().getCircuits()[0].getName()));
    __uint128_t seed = seedMsb;
    seed <<= 64;
    seed += seedLsb;
    std::shared_ptr<CSPRNG> csprng = std::make_shared<ConcreteCSPRNG>(seed);
    OUTCOME_TRY(auto clientProgram,
                ClientProgram::create(programInfo, keyset.client, csprng,
                                      useSimulation));
    OUTCOME_TRY(auto clientCircuit,
                clientProgram.getClientCircuit(
                    programInfo.asReader().getCircuits()[0].getName()));
    auto artifactFolder = std::filesystem::path(sharedLibPath).parent_path();
    return TestCircuit(clientCircuit, serverCircuit, useSimulation,
                       artifactFolder, keyset);
  }

  TestCircuit(ClientCircuit clientCircuit, ServerCircuit serverCircuit,
              bool useSimulation, Keyset keyset)
      : clientCircuit(clientCircuit), serverCircuit(serverCircuit),
        useSimulation(useSimulation), keyset(keyset) {}

  Result<std::vector<Value>> call(std::vector<Value> inputs) {
    auto preparedArgs = std::vector<TransportValue>();
    for (size_t i = 0; i < inputs.size(); i++) {
      OUTCOME_TRY(auto preparedInput, clientCircuit.prepareInput(inputs[i], i));
      preparedArgs.push_back(preparedInput);
    }
    std::vector<TransportValue> returns;
    if (useSimulation) {
      OUTCOME_TRY(returns, serverCircuit.simulate(preparedArgs));
    } else {
      OUTCOME_TRY(returns, serverCircuit.call(keyset.server, preparedArgs));
    }
    std::vector<Value> processedOutputs(returns.size());
    for (size_t i = 0; i < processedOutputs.size(); i++) {
      OUTCOME_TRY(processedOutputs[i],
                  clientCircuit.processOutput(returns[i], i));
    }
    return processedOutputs;
  }

  std::string getArtifactFolder() { return artifactFolder; }

  void setKeySet(Keyset keyset) { this->keyset = keyset; }

private:
  TestCircuit(ClientCircuit clientCircuit, ServerCircuit serverCircuit,
              bool useSimulation, std::string artifactFolder, Keyset keyset)
      : clientCircuit(clientCircuit), serverCircuit(serverCircuit),
        useSimulation(useSimulation), artifactFolder(artifactFolder),
        keyset(keyset){};
  ClientCircuit clientCircuit;
  ServerCircuit serverCircuit;
  bool useSimulation;
  std::string artifactFolder;
  Keyset keyset;
};

TestCircuit load(mlir::concretelang::CompilerEngine::Library compiled) {
  auto keyset =
      getTestKeySetCachePtr()
          ->getKeyset(compiled.getProgramInfo().asReader().getKeyset(), 0, 0)
          .value();
  return TestCircuit::create(
             keyset, compiled.getProgramInfo().asReader(),
             compiled.getSharedLibraryPath(compiled.getOutputDirPath()), 0, 0,
             false)
      .value();
}

const std::string FUNCNAME = "main";

std::string getSystemTempFolderPath() {
  llvm::SmallString<0> tempPath;
  llvm::sys::path::system_temp_directory(true, tempPath);
  return std::string(tempPath);
}

std::string createTempFolderIn(const std::string &rootFolder) {
  std::srand(std::time(nullptr));
  auto new_path = [=]() {
    llvm::SmallString<0> outputPath;
    llvm::sys::path::append(outputPath, rootFolder);
    std::string uid = std::to_string(
        std::hash<std::thread::id>()(std::this_thread::get_id()));
    uid.append("-");
    uid.append(std::to_string(std::rand()));
    llvm::sys::path::append(outputPath, uid);
    return std::string(outputPath);
  };

  // Macos sometimes fail to create new directories. We have to retry a few
  // times.
  for (size_t i = 0; i < 5; i++) {
    auto pathString = new_path();
    auto ec = std::error_code();
    if (!std::filesystem::create_directory(pathString, ec)) {
      std::cout << "Failed to create directory ";
      std::cout << pathString;
      std::cout << " Reason: ";
      std::cout << ec.message();
      std::cout << " Retrying....\n";
      std::cout.flush();
    } else {
      std::cout << "Using artifact folder ";
      std::cout << pathString;
      std::cout << "\n";
      std::cout.flush();
      return pathString;
    }
  }
  std::cout << "Failed to create temp directory 5 times. Aborting...\n";
  std::cout.flush();
  assert(false);
}

void deleteFolder(const std::string &folder) {
  if (!folder.empty()) {
    auto ec = std::error_code();
    if (!std::filesystem::remove_all(folder, ec)) {
      std::cout << "Failed to delete directory ";
      std::cout << folder;
      std::cout << " Reason: ";
      std::cout << ec.message();
      std::cout.flush();
      assert(false);
    }
  }
}

std::vector<uint8_t> values_3bits() { return {0, 1, 2, 5, 7}; }
std::vector<uint8_t> values_6bits() { return {0, 1, 2, 13, 22, 59, 62, 63}; }
std::vector<uint8_t> values_7bits() { return {0, 1, 2, 63, 64, 65, 125, 126}; }

} // namespace testlib
} // namespace concretelang

#endif
