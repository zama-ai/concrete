#include <gtest/gtest.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientLambda.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/TestLib/TestTypedLambda.h"

#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"
#include "tests_tools/keySetCache.h"

testing::Environment *const dfr_env =
    testing::AddGlobalTestEnvironment(new DFREnvironment);

const std::string FUNCNAME = "main";

using namespace concretelang::testlib;

using concretelang::clientlib::scalar_in;
using concretelang::clientlib::scalar_out;
using concretelang::clientlib::tensor1_in;
using concretelang::clientlib::tensor1_out;
using concretelang::clientlib::tensor2_in;
using concretelang::clientlib::tensor2_out;
using concretelang::clientlib::tensor3_out;

std::vector<uint8_t> values_3bits() { return {0, 1, 2, 5, 7}; }
std::vector<uint8_t> values_6bits() { return {0, 1, 2, 13, 22, 59, 62, 63}; }
std::vector<uint8_t> values_7bits() { return {0, 1, 2, 63, 64, 65, 125, 126}; }

mlir::concretelang::CompilerEngine::Library
compile(std::string outputLib, std::string source,
        std::string funcname = FUNCNAME) {
  std::vector<std::string> sources = {source};
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};
  mlir::concretelang::CompilationOptions options(funcname);
  options.emitSDFGOps = true;
#ifdef CONCRETELANG_CUDA_SUPPORT
  options.emitGPUOps = true;
#endif
#ifdef CONCRETELANG_DATAFLOW_TESTING_ENABLED
  options.dataflowParallelize = true;
#endif
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
  auto l = Lambda::load(FUNCNAME, outputLib, 0, 0, getTestKeySetCachePtr());
  assert(l.has_value());
  return l.value();
}

TEST(SDFG_unit_tests, add_eint) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<scalar_out, scalar_in, scalar_in>>(outputLib);
  for (auto a : values_7bits())
    for (auto b : values_7bits()) {
      if (a > b) {
        continue;
      }
      auto res = lambda.call(a, b);
      ASSERT_EQ_OUTCOME(res, (scalar_out)a + b);
    }
}

TEST(SDFG_unit_tests, add_eint_int) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
  %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<scalar_out, scalar_in, scalar_in>>(outputLib);
  for (auto a : values_7bits())
    for (auto b : values_7bits()) {
      if (a > b) {
        continue;
      }
      auto res = lambda.call(a, b);
      ASSERT_EQ_OUTCOME(res, (scalar_out)a + b);
    }
}

TEST(SDFG_unit_tests, mul_eint_int) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
  %1 = "FHE.mul_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<scalar_out, scalar_in, scalar_in>>(outputLib);
  for (auto a : values_3bits())
    for (auto b : values_3bits()) {
      if (a > b) {
        continue;
      }
      auto res = lambda.call(a, b);
      ASSERT_EQ_OUTCOME(res, (scalar_out)a * b);
    }
}

TEST(SDFG_unit_tests, neg_eint) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<scalar_out, scalar_in>>(outputLib);
  for (auto a : values_7bits()) {
    auto res = lambda.call(a);
    ASSERT_EQ_OUTCOME(res, (scalar_out)((a == 0) ? 0 : 256 - a));
  }
}

TEST(SDFG_unit_tests, add_eint_tree) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>, %arg2: !FHE.eint<7>, %arg3: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %2 = "FHE.add_eint"(%arg2, %arg3): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %3 = "FHE.add_eint"(%1, %2): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %3: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<
      TestTypedLambda<scalar_out, scalar_in, scalar_in, scalar_in, scalar_in>>(
      outputLib);
  for (auto a : values_3bits()) {
    for (auto b : values_3bits()) {
      auto res = lambda.call(a, a, b, b);
      ASSERT_EQ_OUTCOME(res, (scalar_out)a + a + b + b);
    }
  }
}

TEST(SDFG_unit_tests, tlu) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
    %tlu_3 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %1 = "FHE.apply_lookup_table"(%arg0, %tlu_3): (!FHE.eint<3>, tensor<8xi64>) -> (!FHE.eint<3>)
    return %1: !FHE.eint<3>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<scalar_out, scalar_in>>(outputLib);
  for (auto a : values_3bits()) {
    auto res = lambda.call(a);
    ASSERT_EQ_OUTCOME(res, (scalar_out)a);
  }
}

TEST(SDFG_unit_tests, tlu_tree) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<4>) -> !FHE.eint<4> {
    %tlu_4 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
    %1 = "FHE.apply_lookup_table"(%arg0, %tlu_4): (!FHE.eint<4>, tensor<16xi64>) -> (!FHE.eint<4>)
    %2 = "FHE.apply_lookup_table"(%arg0, %tlu_4): (!FHE.eint<4>, tensor<16xi64>) -> (!FHE.eint<4>)
    %3 = "FHE.apply_lookup_table"(%1, %tlu_4): (!FHE.eint<4>, tensor<16xi64>) -> (!FHE.eint<4>)
    %4 = "FHE.apply_lookup_table"(%2, %tlu_4): (!FHE.eint<4>, tensor<16xi64>) -> (!FHE.eint<4>)
    %5 = "FHE.add_eint"(%3, %4): (!FHE.eint<4>, !FHE.eint<4>) -> (!FHE.eint<4>)
    %6 = "FHE.apply_lookup_table"(%5, %tlu_4): (!FHE.eint<4>, tensor<16xi64>) -> (!FHE.eint<4>)
    return %6: !FHE.eint<4>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<scalar_out, scalar_in>>(outputLib);
  for (auto a : values_3bits()) {
    auto res = lambda.call(a);
    ASSERT_EQ_OUTCOME(res, (scalar_out)((a * 2) % 16));
  }
}
