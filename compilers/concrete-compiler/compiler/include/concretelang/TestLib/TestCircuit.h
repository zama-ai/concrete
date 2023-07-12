// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TESTLIB_TESTCIRCUIT_H
#define CONCRETELANG_TESTLIB_TESTCIRCUIT_H

#include "boost/outcome.h"

#include "concrete-protocol.pb.h"
#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Values.h"
#include "concretelang/ServerLib/ServerLib.h"
#include <memory>
#include <ostream>

using concretelang::clientlib::ClientCircuit;
using concretelang::clientlib::ClientProgram;
using concretelang::csprng::ConcreteCSPRNG;
using concretelang::error::Result;
using concretelang::keysets::Keyset;
using concretelang::protocol::JSONFileToMessage;
using concretelang::serverlib::ServerCircuit;
using concretelang::serverlib::ServerProgram;
using concretelang::values::TransportValue;
using concretelang::values::Value;

namespace concretelang {
namespace testlib {

class TestCircuit {

public:
  static Result<TestCircuit> create(Keyset keyset,
                                    const concreteprotocol::ProgramInfo &programInfo,
                                    const std::string &sharedLibPath,
                                    uint64_t seedMsb, uint64_t seedLsb,
                                    bool useSimulation = false) {
    OUTCOME_TRY(auto serverProgram,
                ServerProgram::load(programInfo, sharedLibPath, useSimulation));
    OUTCOME_TRY(auto serverCircuit,
                serverProgram.getServerCircuit(programInfo.circuits(0).name()));
    __uint128_t seed = seedMsb;
    seed <<= 64;
    seed += seedLsb;
    auto csprng = ConcreteCSPRNG(seed);
    OUTCOME_TRY(auto clientProgram,
                ClientProgram::create(programInfo, keyset.client, csprng,
                                      useSimulation));
    OUTCOME_TRY(auto clientCircuit,
                clientProgram.getClientCircuit(programInfo.circuits(0).name()));
    auto preparedArgs =
        std::vector<TransportValue>(programInfo.circuits(0).inputs_size());
    return TestCircuit{clientCircuit, serverCircuit, preparedArgs,
                       useSimulation, keyset};
  }

  TestCircuit(ClientCircuit clientCircuit, ServerCircuit serverCircuit,
              std::vector<TransportValue> preparedArgs, bool useSimulation,
              Keyset keyset)
      : clientCircuit(clientCircuit), serverCircuit(serverCircuit),
        preparedArgs(preparedArgs), useSimulation(useSimulation),
        keyset(keyset) {}

  Result<std::vector<Value>> call(std::vector<Value> inputs) {
    for (size_t i = 0; i < preparedArgs.size(); i++) {
      preparedArgs[i] = clientCircuit.prepareInput(inputs[i], i).value();
    }
    std::vector<TransportValue> returns;
    if (useSimulation) {
      returns = serverCircuit.simulate(preparedArgs).value();
    } else {
      auto ret = serverCircuit.call(keyset.server, preparedArgs).as_failure();
      std::cout << ret.error().mesg << std::flush;
      returns = serverCircuit.call(keyset.server, preparedArgs).value();
    }
    std::vector<Value> processedOutputs(returns.size());
    for (size_t i = 0; i < processedOutputs.size(); i++) {
      processedOutputs[i] = clientCircuit.processOutput(returns[i], i).value();
    }
    return processedOutputs;
  }

private:
  ClientCircuit clientCircuit;
  ServerCircuit serverCircuit;
  std::vector<TransportValue> preparedArgs;
  bool useSimulation;
  Keyset keyset;
};

} // namespace testlib
} // namespace concretelang

#endif
