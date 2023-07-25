#include <cstdint>
#include <gtest/gtest.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

#include "boost/outcome.h"

#include "concrete-protocol.pb.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/TestLib/TestCircuit.h"

#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"
#include "tests_tools/keySetCache.h"

testing::Environment *const dfr_env =
    testing::AddGlobalTestEnvironment(new DFREnvironment);

const std::string FUNCNAME = "main";

using namespace concretelang::testlib;
namespace encodings = mlir::concretelang::encodings;

mlir::concretelang::CompilerEngine::Library
compile(std::string outputLib, std::string source,
        std::string funcname = FUNCNAME) {
  std::vector<std::string> sources = {source};
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};
  mlir::concretelang::CompilationOptions options(funcname);

  auto encoding = concreteprotocol::EncodingInfo();
  encoding.set_allocated_shape(new concreteprotocol::Shape());
  auto integer = new concreteprotocol::IntegerCiphertextEncodingInfo();
  integer->set_allocated_native(
      new concreteprotocol::IntegerCiphertextEncodingInfo::NativeMode());
  integer->set_width(3);
  integer->set_issigned(false);
  encoding.set_allocated_integerciphertext(integer);
  options.encodings = concreteprotocol::CircuitEncodingInfo();
  options.encodings->mutable_inputs()->AddAllocated(
      new concreteprotocol::EncodingInfo(encoding));
  options.encodings->mutable_inputs()->AddAllocated(
      new concreteprotocol::EncodingInfo(encoding));
  options.encodings->mutable_outputs()->AddAllocated(
      new concreteprotocol::EncodingInfo(encoding));
  options.encodings->set_allocated_name(new std::string("main"));
  options.v0Parameter = {2, 10, 693, 4, 9, 7, 2, std::nullopt};
  ce.setCompilationOptions(options);
  auto result = ce.compile(sources, outputLib);
  if (!result) {
    llvm::errs() << result.takeError();
    assert(false);
  }
  assert(result);
  return result.get();
}

static const std::string CURRENT_FILE = __FILE__;
static const std::string THIS_TEST_DIRECTORY =
    CURRENT_FILE.substr(0, CURRENT_FILE.find_last_of("/\\"));
static const std::string OUT_DIRECTORY = "/tmp";

template <typename Info> std::string outputLibFromThis(Info *info) {
  return OUT_DIRECTORY + "/" + std::string(info->name());
}

TestCircuit load(mlir::concretelang::CompilerEngine::Library compiled) {
  auto keyset = getTestKeySetCachePtr()
                    ->getKeyset(compiled.getProgramInfo().keyset(), 0, 0)
                    .value();
  return TestCircuit::create(keyset, compiled.getProgramInfo(),
                             compiled.getOutputDirPath(), 0, 0, false)
      .value();
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
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto circuit = load(compile(outputLib, source));
  uint64_t a = 5;
  uint64_t b = 5;
  auto res = circuit.call({
    Tensor<uint64_t>(a),
    Tensor<uint64_t>(b)
  });
  ASSERT_TRUE(res.has_value());
  auto out = res.value()[0].getTensor<uint64_t>()->values[0];
  ASSERT_EQ(out, a + b);
}
