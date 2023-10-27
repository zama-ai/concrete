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
#include "concretelang/TestLib/TestCircuit.h"

#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"

testing::Environment *const dfr_env =
    testing::AddGlobalTestEnvironment(new DFREnvironment);

using namespace concretelang::testlib;
namespace encodings = mlir::concretelang::encodings;

mlir::concretelang::CompilerEngine::Library
compile(std::string artifactFolder, std::string source,
        std::string funcname = FUNCNAME) {
  std::vector<std::string> sources = {source};
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};
  mlir::concretelang::CompilationOptions options(funcname);

  options.encodings = Message<concreteprotocol::CircuitEncodingInfo>();
  auto inputs = options.encodings->asBuilder().initInputs(2);
  auto outputs = options.encodings->asBuilder().initOutputs(1);

  auto encodingInfo = Message<concreteprotocol::EncodingInfo>().asBuilder();
  encodingInfo.initShape();
  auto integer = encodingInfo.getEncoding().initIntegerCiphertext();
  integer.getMode().initNative();
  integer.setWidth(3);
  integer.setIsSigned(false);

  inputs.setWithCaveats(0, encodingInfo);
  inputs.setWithCaveats(1, encodingInfo);
  outputs.setWithCaveats(0, encodingInfo);

  options.encodings->asBuilder().setName("main");
  options.v0Parameter = {2, 10, 693, 4, 9, 7, 2, std::nullopt};
  ce.setCompilationOptions(options);
  auto result = ce.compile(sources, artifactFolder);
  if (!result) {
    llvm::errs() << result.takeError();
    assert(false);
  }
  assert(result);
  return result.get();
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
  std::string artifactFolder = createTempFolderIn(getSystemTempFolderPath());
  auto circuit = load(compile(artifactFolder, source));
  uint64_t a = 5;
  uint64_t b = 5;
  auto res = circuit.call({Tensor<uint64_t>(a), Tensor<uint64_t>(b)});
  ASSERT_TRUE(res.has_value());
  auto out = res.value()[0].getTensor<uint64_t>()->values[0];
  ASSERT_EQ(out, a + b);
  deleteFolder(artifactFolder);
}
