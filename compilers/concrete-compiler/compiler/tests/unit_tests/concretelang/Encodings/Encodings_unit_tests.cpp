#include <gtest/gtest.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientLambda.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/TestLib/TestTypedLambda.h"

#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"
#include "tests_tools/keySetCache.h"

testing::Environment *const dfr_env =
    testing::AddGlobalTestEnvironment(new DFREnvironment);

const std::string FUNCNAME = "main";

using namespace concretelang::testlib;
namespace encodings = mlir::concretelang::encodings;
using concretelang::clientlib::scalar_in;
using concretelang::clientlib::scalar_out;
using concretelang::clientlib::tensor1_in;
using concretelang::clientlib::tensor1_out;
using concretelang::clientlib::tensor2_in;
using concretelang::clientlib::tensor2_out;
using concretelang::clientlib::tensor3_out;

mlir::concretelang::CompilerEngine::Library
compile(std::string outputLib, std::string source,
        std::string funcname = FUNCNAME) {
  std::vector<std::string> sources = {source};
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};
  mlir::concretelang::CompilationOptions options(funcname);
  options.encodings = encodings::CircuitEncodings{
      {
          encodings::EncryptedIntegerScalarEncoding{3, false},
          encodings::EncryptedIntegerScalarEncoding{3, false},
      },
      {
          encodings::EncryptedIntegerScalarEncoding{3, false},
      }};
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

template <typename Lambda> Lambda load(std::string outputLib) {
  auto l =
      Lambda::load(FUNCNAME, outputLib, 0, 0, 0, 0, getTestKeySetCachePtr());
  assert(l.has_value());
  return l.value();
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
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<scalar_out, scalar_in, scalar_in>>(outputLib);
  scalar_in a = 5;
  scalar_in b = 5;
  auto res = lambda.call(a, b);
  ASSERT_EQ_OUTCOME(res, (scalar_out)a + b);
}
