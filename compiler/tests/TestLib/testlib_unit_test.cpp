#include <gtest/gtest.h>

#include <numeric>
#include <cassert>

#include "../unittest/end_to_end_jit_test.h"
#include "concretelang/TestLib/DynamicLambda.h"

const std::string FUNCNAME = "main";

template<typename... Params>
using TypedDynamicLambda = mlir::concretelang::TypedDynamicLambda<Params...>;

using scalar = uint64_t;
using tensor1_in = std::vector<uint8_t>;
using tensor1_out = std::vector<uint64_t>;
using tensor2_out = std::vector<std::vector<uint64_t>>;
using tensor3_out = std::vector<std::vector<std::vector<uint64_t>>>;

std::vector<uint8_t>
values_7bits() {
  return {0, 1, 2, 63, 64, 65, 125, 126};
}

llvm::Expected<mlir::concretelang::CompilerEngine::Library>
compile(std::string outputLib, std::string source) {
  std::vector<std::string> sources = {source};
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::JitCompilerEngine ce {ccx};
  ce.setClientParametersFuncName(FUNCNAME);
  return ce.compile(sources, outputLib);
}

template<typename Info>
std::string outputLibFromThis(Info *info) {
  return "tests/TestLib/out/" + std::string(info->name());
}

TEST(CompiledModule, call_1s_1s) {
  std::string source = R"(
func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  return %arg0: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<scalar, scalar>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  for(auto a: values_7bits()) {
    auto res = lambda->call(a);
    ASSERT_EXPECTED_VALUE(res, a);
  }
}

TEST(CompiledModule, call_2s_1s) {
  std::string source = R"(
func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<scalar, scalar, scalar>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  for(auto a: values_7bits()) for(auto b: values_7bits()) {
    auto res = lambda->call(a, b);
    ASSERT_EXPECTED_VALUE(res, a + b);
  }
}

TEST(CompiledModule, call_1s_1t) {
  std::string source = R"(
func @main(%arg0: !FHE.eint<7>) -> tensor<1x!FHE.eint<7>> {
  %1 = tensor.from_elements %arg0 : tensor<1x!FHE.eint<7>>
  return %1: tensor<1x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<tensor1_out, scalar>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  for(auto a: values_7bits()) {
    auto res = lambda->call(a);
    ASSERT_EXPECTED_SUCCESS(res);
    tensor1_out v = res.get();
    EXPECT_EQ(v[0], a);
  }
}

TEST(CompiledModule, call_2s_1t) {
  std::string source = R"(
func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> tensor<2x!FHE.eint<7>> {
  %1 = tensor.from_elements %arg0, %arg1 : tensor<2x!FHE.eint<7>>
  return %1: tensor<2x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<tensor1_out, scalar, scalar>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  for(auto a : values_7bits()) {
    auto res = lambda->call(a, a+1);
    ASSERT_EXPECTED_SUCCESS(res);
    tensor1_out v = res.get();
    EXPECT_EQ((scalar)v[0], a);
    EXPECT_EQ((scalar)v[1], a + 1u);
  }
}

TEST(CompiledModule, call_1t_1s) {
  std::string source = R"(
func @main(%arg0: tensor<1x!FHE.eint<7>>) -> !FHE.eint<7> {
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %arg0[%c0] : tensor<1x!FHE.eint<7>>
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<scalar, tensor1_in>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  for(uint8_t a : values_7bits()) {
    tensor1_in ta = {a};
    auto res = lambda->call(ta);
    ASSERT_EXPECTED_VALUE(res, a);
  }
}

TEST(CompiledModule, call_1t_1t) {
  std::string source = R"(
func @main(%arg0: tensor<3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
  return %arg0: tensor<3x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<tensor1_out, tensor1_in>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  tensor1_in ta = {1, 2, 3};
  auto res = lambda->call(ta);
  ASSERT_EXPECTED_SUCCESS(res);
  tensor1_out v = res.get();
  for(size_t i = 0; i < v.size(); i++) {
    EXPECT_EQ(v[i], ta[i]);
  }
}

TEST(CompiledModule, call_2t_1s) {
  std::string source = R"(
func @main(%arg0: tensor<3x!FHE.eint<7>>, %arg1: tensor<3x!FHE.eint<7>>) -> !FHE.eint<7> {
  %1 = "FHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x!FHE.eint<7>>, tensor<3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>
  %c1 = arith.constant 1 : i8
  %2 = tensor.from_elements %c1, %c1, %c1 : tensor<3xi8>
  %3 = "FHELinalg.dot_eint_int"(%1, %2) : (tensor<3x!FHE.eint<7>>, tensor<3xi8>) -> !FHE.eint<7>
  return %3: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<scalar, tensor1_in, std::array<uint8_t, 3>>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  tensor1_in ta {1, 2, 3};
  std::array<uint8_t, 3> tb {5, 7, 9};
  auto res = lambda->call(ta, tb);
  auto expected = std::accumulate(ta.begin(), ta.end(), 0u) +
    std::accumulate(tb.begin(), tb.end(), 0u);
  ASSERT_EXPECTED_VALUE(res, expected);
}

TEST(CompiledModule, call_1tr2_1tr2) {
  std::string source = R"(
func @main(%arg0: tensor<2x3x!FHE.eint<7>>) -> tensor<2x3x!FHE.eint<7>> {
  return %arg0: tensor<2x3x!FHE.eint<7>>
}
)";
  using tensor2_in = std::array<std::array<uint8_t, 3>, 2>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<tensor2_out, tensor2_in>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  tensor2_in ta = {{
    {1, 2, 3},
    {4, 5, 6}
  }};
  auto res = lambda->call(ta);
  ASSERT_EXPECTED_SUCCESS(res);
  tensor2_out v = res.get();
  for(size_t i = 0; i < v.size(); i++) {
    for(size_t j = 0; j < v.size(); j++) {
      EXPECT_EQ(v[i][j], ta[i][j]);
    }
  }
}


TEST(CompiledModule, call_1tr3_1tr3) {
  std::string source = R"(
func @main(%arg0: tensor<2x3x1x!FHE.eint<7>>) -> tensor<2x3x1x!FHE.eint<7>> {
  return %arg0: tensor<2x3x1x!FHE.eint<7>>
}
)";
  using tensor3_in = std::array<std::array<std::array<uint8_t, 1>, 3>, 2>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  ASSERT_EXPECTED_SUCCESS(compiled);
  auto lambda = TypedDynamicLambda<tensor3_out, tensor3_in>::load(FUNCNAME, outputLib);
  ASSERT_EXPECTED_SUCCESS(lambda);
  ASSERT_LLVM_ERROR(lambda->generateKeySet(getTestKeySetCache()));
  tensor3_in ta = {{
    {{ {1}, {2}, {3} }},
    {{ {4}, {5}, {6} }}
  }};
  auto res = lambda->call(ta);
  ASSERT_EXPECTED_SUCCESS(res);
  tensor3_out v = res.get();
  for(size_t i = 0; i < v.size(); i++) {
    for(size_t j = 0; j < v[i].size(); j++) {
      for(size_t k = 0; k < v[i][j].size(); k++) {
        EXPECT_EQ(v[i][j][k], ta[i][j][k]);
      }
    }
  }
}
