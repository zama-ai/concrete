#include <cstdint>
#include <gtest/gtest.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

#include "boost/outcome.h"

#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/TestLib/TestProgram.h"

#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"

testing::Environment *const dfr_env =
    testing::AddGlobalTestEnvironment(new DFREnvironment);

using namespace concretelang::testlib;
namespace encodings = mlir::concretelang::encodings;

Result<TestProgram> setupTestProgram(std::string source,
                                     std::string funcname = FUNCNAME) {
  std::vector<std::string> sources = {source};
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};
  mlir::concretelang::CompilationOptions options;

  auto circuitEncoding = Message<concreteprotocol::CircuitEncodingInfo>();
  auto inputs = circuitEncoding.asBuilder().initInputs(2);
  auto outputs = circuitEncoding.asBuilder().initOutputs(1);
  circuitEncoding.asBuilder().setName(funcname);

  auto encodingInfo = Message<concreteprotocol::EncodingInfo>();
  encodingInfo.asBuilder().initShape();
  auto integer = encodingInfo.asBuilder().getEncoding().initIntegerCiphertext();
  integer.getMode().initNative();
  integer.setWidth(3);
  integer.setIsSigned(false);

  inputs.setWithCaveats(0, encodingInfo.asReader());
  inputs.setWithCaveats(1, encodingInfo.asReader());
  outputs.setWithCaveats(0, encodingInfo.asReader());

  options.encodings = Message<concreteprotocol::ProgramEncodingInfo>();
  options.encodings->asBuilder().initCircuits(1).setWithCaveats(
      0, circuitEncoding.asReader());

  options.v0Parameter = {2, 10, 693, 4, 9, 7, 2, std::nullopt};
  TestProgram testCircuit(options);
  OUTCOME_TRYV(testCircuit.compile({source}));
  OUTCOME_TRYV(testCircuit.generateKeyset());
  return std::move(testCircuit);
}

TEST(Encodings_unit_tests, multi_key) {
  std::string source = R"(
func.func @main(
  %arg0: !TFHE.glwe<sk<1,1,2048>>,
  %arg1: !TFHE.glwe<sk<2,1,2048>>
  ) -> !TFHE.glwe<sk<2,1,2048>> {

  %0 = "TFHE.keyswitch_glwe"(%arg0) {key=#TFHE.ksk<sk<1,1,2048>, sk<2, 1,2048>, 7, 2>} : (!TFHE.glwe<sk<1, 1, 2048>>) -> !TFHE.glwe<sk<2, 1, 2048>>
  %1 = "TFHE.add_glwe"(%arg1, %0) : (!TFHE.glwe<sk<2,1,2048>>, !TFHE.glwe<sk<2,1,2048>>) -> !TFHE.glwe<sk<2,1,2048>>
  return %1 : !TFHE.glwe<sk<2,1,2048>>

}
)";
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  uint64_t a = 5;
  uint64_t b = 5;
  auto res = circuit.call({Tensor<uint64_t>(a), Tensor<uint64_t>(b)});
  ASSERT_TRUE(res.has_value());
  auto out = res.value()[0].getTensor<uint64_t>()->values[0];
  ASSERT_EQ(out, a + b);
}
