// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TESTLIB_TESTCIRCUIT_H
#define CONCRETELANG_TESTLIB_TESTCIRCUIT_H

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Values.h"
#include "concretelang/ServerLib/ServerLib.h"
#include <memory>

using concretelang::values::Value;
using concretelang::keysets::Keyset;
using concretelang::clientlib::ClientCircuit;
using concretelang::clientlib::ClientProgram;
using concretelang::serverlib::ServerCircuit;
using concretelang::serverlib::ServerProgram;


namespace concretelang {
namespace testlib {

class TestCircuit{

    TestCircuit(std::shared_ptr<keyset> keyset, const concreteprotocol::ProgramInfo &programInfo, std::string &sharedLibPath, bool useSimulation=false){
        auto serverProgram = ServerProgram::load(programInfo, sharedLibPath, useSimulation).value();
        auto serverClient = serverProgram.getCircuit(programInfo.circuits(0).name()).value();
    }

    std::vector<Value> call(std::vector<Value> inputs){

    }

private:
  ClientCircuit clientCircuit;
  ServerCircuit serverCircuit;
  bool useSimulation;
  std::shared_ptr<Keyset> keyset;
}

} // namespace testlib
} // namespace concretelang

#endif
