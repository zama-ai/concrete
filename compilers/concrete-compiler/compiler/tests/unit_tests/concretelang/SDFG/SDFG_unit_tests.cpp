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
#ifdef CONCRETELANG_CUDA_SUPPORT
  options.emitGPUOps = true;
  // FIXME(#71)
#ifdef __APPLE__
  options.emitSDFGOps = false;
#else
  options.emitSDFGOps = true;
#endif
#endif
  options.batchTFHEOps = true;
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

TEST(SDFG_unit_tests, tlu_batched) {
  std::string source = R"(
    func.func @main(%t: tensor<3x3x!FHE.eint<2>>) -> tensor<3x3x!FHE.eint<3>> {
      %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
      %res = "FHELinalg.apply_lookup_table"(%t, %lut) : (tensor<3x3x!FHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!FHE.eint<3>>
      return %res : tensor<3x3x!FHE.eint<3>>
    }
)";
  using tensor2_in = std::array<std::array<uint8_t, 3>, 3>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<tensor2_out, tensor2_in>>(outputLib);
  tensor2_in t = {{{0, 1, 2}, {3, 0, 1}, {2, 3, 0}}};
  tensor2_out expected = {{{1, 3, 5}, {7, 1, 3}, {5, 7, 1}}};
  auto res = lambda.call(t);
  ASSERT_TRUE(res);
  ASSERT_EQ_OUTCOME(res, expected);
}

TEST(SDFG_unit_tests, batched_tree) {
  std::string source = R"(
    func.func @main(%t: tensor<3x3x!FHE.eint<3>>, %a1: tensor<3x3xi4>, %a2: tensor<3x3xi4>) -> tensor<3x3x!FHE.eint<4>> {
      %lut = arith.constant dense<[1,3,5,7,9,11,13,15]> : tensor<8xi64>
      %b1 = "FHELinalg.add_eint_int"(%t, %a1) : (tensor<3x3x!FHE.eint<3>>, tensor<3x3xi4>) -> tensor<3x3x!FHE.eint<3>>
      %b2 = "FHELinalg.add_eint_int"(%t, %a2) : (tensor<3x3x!FHE.eint<3>>, tensor<3x3xi4>) -> tensor<3x3x!FHE.eint<3>>
      %c = "FHELinalg.add_eint"(%b1, %b2) : (tensor<3x3x!FHE.eint<3>>, tensor<3x3x!FHE.eint<3>>) -> tensor<3x3x!FHE.eint<3>>
      %res = "FHELinalg.apply_lookup_table"(%c, %lut) : (tensor<3x3x!FHE.eint<3>>, tensor<8xi64>) -> tensor<3x3x!FHE.eint<4>>
      return %res : tensor<3x3x!FHE.eint<4>>
    }
)";
  using tensor2_in = std::array<std::array<uint8_t, 3>, 3>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<tensor2_out, tensor2_in, tensor2_in, tensor2_in>>(
          outputLib);
  tensor2_in t = {{{0, 1, 2}, {3, 0, 1}, {2, 3, 0}}};
  tensor2_in a1 = {{{0, 1, 0}, {0, 1, 0}, {0, 1, 0}}};
  tensor2_in a2 = {{{1, 0, 1}, {1, 0, 1}, {1, 0, 1}}};
  tensor2_out expected = {{{3, 7, 11}, {15, 3, 7}, {11, 15, 3}}};
  auto res = lambda.call(t, a1, a2);
  ASSERT_TRUE(res);
  ASSERT_EQ_OUTCOME(res, expected);
}

TEST(SDFG_unit_tests, batched_tree_mapped_tlu) {
  std::string source = R"(
    func.func @main(%t: tensor<3x3x!FHE.eint<3>>, %a1: tensor<3x3xi4>, %a2: tensor<3x3xi4>) -> tensor<3x3x!FHE.eint<4>> {
      %lut_vec = arith.constant dense<[[1,3,5,7,9,11,13,15],
                                       [2,4,6,8,10,12,14,0],
                                       [3,6,9,12,15,2,5,8],
                                       [4,8,12,0,4,8,12,0]]> : tensor<4x8xi64>
      %map = arith.constant dense<[[0, 1, 2], [3, 2, 1], [1, 2, 3]]> : tensor<3x3xindex>
      %b1 = "FHELinalg.add_eint_int"(%t, %a1) : (tensor<3x3x!FHE.eint<3>>, tensor<3x3xi4>) -> tensor<3x3x!FHE.eint<3>>
      %b2 = "FHELinalg.add_eint_int"(%t, %a2) : (tensor<3x3x!FHE.eint<3>>, tensor<3x3xi4>) -> tensor<3x3x!FHE.eint<3>>
      %c = "FHELinalg.add_eint"(%b1, %b2) : (tensor<3x3x!FHE.eint<3>>, tensor<3x3x!FHE.eint<3>>) -> tensor<3x3x!FHE.eint<3>>
      %res = "FHELinalg.apply_mapped_lookup_table"(%c, %lut_vec, %map) : (tensor<3x3x!FHE.eint<3>>, tensor<4x8xi64>, tensor<3x3xindex>) -> tensor<3x3x!FHE.eint<4>>
      return %res : tensor<3x3x!FHE.eint<4>>
    }
)";
  using tensor2_in = std::array<std::array<uint8_t, 3>, 3>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<tensor2_out, tensor2_in, tensor2_in, tensor2_in>>(
          outputLib);
  tensor2_in t = {{{0, 1, 2}, {3, 0, 1}, {2, 3, 0}}};
  tensor2_in a1 = {{{0, 1, 0}, {0, 1, 0}, {0, 1, 0}}};
  tensor2_in a2 = {{{1, 0, 1}, {1, 0, 1}, {1, 0, 1}}};
  tensor2_out expected = {{{3, 8, 2}, {0, 6, 8}, {12, 8, 8}}};
  auto res = lambda.call(t, a1, a2);
  ASSERT_TRUE(res);
  ASSERT_EQ_OUTCOME(res, expected);
}
