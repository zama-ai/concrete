#include <gtest/gtest.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <ostream>
#include <thread>

#include "boost/outcome.h"

#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/TestLib/TestProgram.h"

#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"

using namespace concretelang::testlib;

testing::Environment *const dfr_env =
    testing::AddGlobalTestEnvironment(new DFREnvironment);

Result<TestProgram> setupTestProgram(std::string source,
                                     std::string funcname = FUNCNAME) {
  mlir::concretelang::CompilationOptions options;
#ifdef CONCRETELANG_CUDA_SUPPORT
  options.emitGPUOps = true;
  options.emitSDFGOps = true;
#endif
  options.batchTFHEOps = true;
  TestProgram testCircuit(options);
  OUTCOME_TRYV(testCircuit.compile({source}));
  OUTCOME_TRYV(testCircuit.generateKeyset());
  return std::move(testCircuit);
}

TEST(SDFG_unit_tests, add_eint) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  for (auto a : values_7bits())
    for (auto b : values_7bits()) {
      if (a > b) {
        continue;
      }
      auto res = circuit.call({Tensor<uint64_t>(a), Tensor<uint64_t>(b)});
      ASSERT_TRUE(res.has_value());
      auto out = res.value()[0].getTensor<uint64_t>().value()[0];
      ASSERT_EQ(out, (uint64_t)a + b);
    }
}

TEST(SDFG_unit_tests, add_eint_int) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
  %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  for (auto a : values_7bits())
    for (auto b : values_7bits()) {
      if (a > b) {
        continue;
      }
      auto res = circuit.call({Tensor<uint64_t>(a), Tensor<uint8_t>(b)});
      ASSERT_TRUE(res.has_value());
      auto out = res.value()[0].getTensor<uint64_t>().value()[0];
      ASSERT_EQ(out, (uint64_t)a + b);
    }
}

TEST(SDFG_unit_tests, mul_eint_int) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
  %1 = "FHE.mul_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  for (auto a : values_3bits())
    for (auto b : values_3bits()) {
      if (a > b) {
        continue;
      }
      auto res = circuit.call({Tensor<uint64_t>(a), Tensor<uint8_t>(b)});
      ASSERT_TRUE(res.has_value());
      auto out = res.value()[0].getTensor<uint64_t>().value()[0];
      ASSERT_EQ(out, (uint64_t)a * b);
    }
}

TEST(SDFG_unit_tests, neg_eint) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  for (auto a : values_7bits()) {
    auto res = circuit.call({Tensor<uint64_t>(a)});
    ASSERT_TRUE(res.has_value());
    auto out = res.value()[0].getTensor<uint64_t>().value()[0];
    ASSERT_EQ(out, (uint64_t)((a == 0) ? 0 : 256 - a));
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
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  for (auto a : values_3bits()) {
    for (auto b : values_3bits()) {
      auto res = circuit.call({
          Tensor<uint64_t>(a),
          Tensor<uint64_t>(a),
          Tensor<uint64_t>(b),
          Tensor<uint64_t>(b),
      });
      ASSERT_TRUE(res.has_value());
      auto out = res.value()[0].getTensor<uint64_t>().value()[0];
      ASSERT_EQ(out, (uint64_t)a + a + b + b);
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
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  for (auto a : values_3bits()) {
    auto res = circuit.call({Tensor<uint64_t>(a)});
    ASSERT_TRUE(res.has_value());
    auto out = res.value()[0].getTensor<uint64_t>().value()[0];
    ASSERT_EQ(out, (uint64_t)a);
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
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  for (auto a : values_3bits()) {
    auto res = circuit.call({Tensor<uint64_t>(a)});
    ASSERT_TRUE(res.has_value());
    auto out = res.value()[0].getTensor<uint64_t>().value()[0];
    ASSERT_EQ(out, (uint64_t)((a * 2) % 16));
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
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  auto t = Tensor<uint64_t>({0, 1, 2, 3, 0, 1, 2, 3, 0}, {3, 3});
  auto expected = Tensor<uint64_t>({1, 3, 5, 7, 1, 3, 5, 7, 1}, {3, 3});
  auto res = circuit.call({t});
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value()[0].getTensor<uint64_t>(), expected);
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
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  auto t = Tensor<uint64_t>({0, 1, 2, 3, 0, 1, 2, 3, 0}, {3, 3});
  auto a1 = Tensor<uint8_t>({0, 1, 0, 0, 1, 0, 0, 1, 0}, {3, 3});
  auto a2 = Tensor<uint8_t>({1, 0, 1, 1, 0, 1, 1, 0, 1}, {3, 3});
  auto expected = Tensor<uint64_t>({3, 7, 11, 15, 3, 7, 11, 15, 3}, {3, 3});
  auto res = circuit.call({t, a1, a2});
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value()[0].getTensor<uint64_t>(), expected);
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
  ASSERT_ASSIGN_OUTCOME_VALUE(circuit, setupTestProgram(source));
  auto t = Tensor<uint64_t>({0, 1, 2, 3, 0, 1, 2, 3, 0}, {3, 3});
  auto a1 = Tensor<uint8_t>({0, 1, 0, 0, 1, 0, 0, 1, 0}, {3, 3});
  auto a2 = Tensor<uint8_t>({1, 0, 1, 1, 0, 1, 1, 0, 1}, {3, 3});
  auto expected = Tensor<uint64_t>({3, 8, 2, 0, 6, 8, 12, 8, 8}, {3, 3});
  auto res = circuit.call({t, a1, a2});
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value()[0].getTensor<uint64_t>(), expected);
}
