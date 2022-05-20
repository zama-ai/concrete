#include <gtest/gtest.h>

#include <cassert>
#include <fstream>
#include <numeric>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientLambda.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/TestLib/TestTypedLambda.h"

#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"
#include "tests_tools/keySetCache.h"

#include "call_2t_1s_with_header-client.h.generated"

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
#ifdef CONCRETELANG_PARALLEL_TESTING_ENABLED
  options.autoParallelize = true;
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

TEST(CompiledModule, call_1s_1s_client_view) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  return %arg0: !FHE.eint<7>
}
)";
  namespace clientlib = concretelang::clientlib;
  using MyLambda = clientlib::TypedClientLambda<scalar_out, scalar_in>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  std::string jsonPath = compiled.getClientParametersPath(outputLib);
  auto maybeLambda = MyLambda::load("main", jsonPath);
  ASSERT_TRUE(maybeLambda.has_value());
  auto lambda = maybeLambda.value();
  auto maybeKeySet = lambda.keySet(getTestKeySetCachePtr(), 0, 0);
  ASSERT_TRUE(maybeKeySet.has_value());
  std::shared_ptr<KeySet> keySet = std::move(maybeKeySet.value());
  auto maybePublicArguments = lambda.publicArguments(1, *keySet);

  ASSERT_TRUE(maybePublicArguments.has_value());
  auto publicArguments = std::move(maybePublicArguments.value());
  std::ostringstream osstream(std::ios::binary);
  ASSERT_TRUE(publicArguments->serialize(osstream).has_value());
  EXPECT_TRUE(osstream.good());
  // Direct call without intermediate
  EXPECT_TRUE(lambda.serializeCall(1, *keySet, osstream));
  EXPECT_TRUE(osstream.good());
}

TEST(CompiledModule, call_1s_1s) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  return %arg0: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<scalar_out, scalar_in>>(outputLib);
  for (auto a : values_7bits()) {
    auto res = lambda.call(a);
    ASSERT_EQ_OUTCOME(res, a);
  }
}

TEST(CompiledModule, call_2s_1s_choose) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  return %arg0: !FHE.eint<7>
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
      ASSERT_EQ_OUTCOME(res, a);
    }
}

TEST(CompiledModule, call_2s_1s) {

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
      ASSERT_EQ_OUTCOME(res, (int64_t)a + b);
    }
}

TEST(CompiledModule, call_1s_1s_bad_call) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<scalar_out, scalar_in>>(outputLib);
  auto res = lambda.call(1);
  ASSERT_FALSE(res.has_value());
}

TEST(CompiledModule, call_1s_1t) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>) -> tensor<1x!FHE.eint<7>> {
  %1 = tensor.from_elements %arg0 : tensor<1x!FHE.eint<7>>
  return %1: tensor<1x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<tensor1_out, scalar_in>>(outputLib);
  for (auto a : values_7bits()) {
    auto res = lambda.call(a);
    EXPECT_TRUE(res);
    tensor1_out v = res.value();
    EXPECT_EQ(v[0], a);
  }
}

TEST(CompiledModule, call_2s_1t) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> tensor<2x!FHE.eint<7>> {
  %1 = tensor.from_elements %arg0, %arg1 : tensor<2x!FHE.eint<7>>
  return %1: tensor<2x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<tensor1_out, scalar_in, scalar_in>>(outputLib);
  for (auto a : values_7bits()) {
    auto res = lambda.call(a, a + 1);
    EXPECT_TRUE(res);
    tensor1_out v = res.value();
    EXPECT_EQ(v[0], (scalar_out)a);
    EXPECT_EQ(v[1], (scalar_out)(a + 1u));
  }
}

TEST(CompiledModule, call_1t_1s) {
  std::string source = R"(
func.func @main(%arg0: tensor<1x!FHE.eint<7>>) -> !FHE.eint<7> {
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %arg0[%c0] : tensor<1x!FHE.eint<7>>
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<scalar_out, tensor1_in>>(outputLib);
  for (uint8_t a : values_7bits()) {
    tensor1_in ta = {a};
    auto res = lambda.call(ta);
    ASSERT_EQ_OUTCOME(res, a);
  }
}

TEST(CompiledModule, call_1t_1t) {
  std::string source = R"(
func.func @main(%arg0: tensor<3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
  return %arg0: tensor<3x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<tensor1_out, tensor1_in>>(outputLib);
  tensor1_in ta = {1, 2, 3};
  auto res = lambda.call(ta);
  ASSERT_TRUE(res);
  tensor1_out v = res.value();
  for (size_t i = 0; i < v.size(); i++) {
    EXPECT_EQ(v[i], ta[i]);
  }
}

TEST(CompiledModule, call_2t_1s) {
  std::string source = R"(
func.func @main(%arg0: tensor<3x!FHE.eint<7>>, %arg1: tensor<3x!FHE.eint<7>>) -> !FHE.eint<7> {
  %1 = "FHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x!FHE.eint<7>>, tensor<3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>
  %c1 = arith.constant 1 : i8
  %2 = tensor.from_elements %c1, %c1, %c1 : tensor<3xi8>
  %3 = "FHELinalg.dot_eint_int"(%1, %2) : (tensor<3x!FHE.eint<7>>, tensor<3xi8>) -> !FHE.eint<7>
  return %3: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<scalar_out, tensor1_in, std::array<uint8_t, 3>>>(
          outputLib);
  tensor1_in ta{1, 2, 3};
  std::array<uint8_t, 3> tb{5, 7, 9};
  auto res = lambda.call(ta, tb);
  auto expected = std::accumulate(ta.begin(), ta.end(), 0u) +
                  std::accumulate(tb.begin(), tb.end(), 0u);
  ASSERT_EQ_OUTCOME(res, expected);
}

TEST(CompiledModule, call_1tr2_1tr2) {
  std::string source = R"(
func.func @main(%arg0: tensor<2x3x!FHE.eint<7>>) -> tensor<2x3x!FHE.eint<7>> {
  return %arg0: tensor<2x3x!FHE.eint<7>>
}
)";
  using tensor2_in = std::array<std::array<uint8_t, 3>, 2>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<tensor2_out, tensor2_in>>(outputLib);
  tensor2_in ta = {{{1, 2, 3}, {4, 5, 6}}};
  auto res = lambda.call(ta);
  ASSERT_TRUE(res);
  tensor2_out v = res.value();
  for (size_t i = 0; i < v.size(); i++) {
    for (size_t j = 0; j < v.size(); j++) {
      EXPECT_EQ(v[i][j], ta[i][j]);
    }
  }
}

TEST(CompiledModule, call_1tr3_1tr3) {
  std::string source = R"(
func.func @main(%arg0: tensor<2x3x1x!FHE.eint<7>>) -> tensor<2x3x1x!FHE.eint<7>> {
  return %arg0: tensor<2x3x1x!FHE.eint<7>>
}
)";
  using tensor3_in = std::array<std::array<std::array<uint8_t, 1>, 3>, 2>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda = load<TestTypedLambda<tensor3_out, tensor3_in>>(outputLib);
  tensor3_in ta = {{{{{1}, {2}, {3}}}, {{{4}, {5}, {6}}}}};
  auto res = lambda.call(ta);
  ASSERT_TRUE(res);
  tensor3_out v = res.value();
  for (size_t i = 0; i < v.size(); i++) {
    for (size_t j = 0; j < v[i].size(); j++) {
      for (size_t k = 0; k < v[i][j].size(); k++) {
        EXPECT_EQ(v[i][j][k], ta[i][j][k]);
      }
    }
  }
}

TEST(CompiledModule, call_2tr3_1tr3) {
  std::string source = R"(
func.func @main(%arg0: tensor<2x3x1x!FHE.eint<7>>, %arg1: tensor<2x3x1x!FHE.eint<7>>) -> tensor<2x3x1x!FHE.eint<7>> {
  %1 = "FHELinalg.add_eint"(%arg0, %arg1): (tensor<2x3x1x!FHE.eint<7>>, tensor<2x3x1x!FHE.eint<7>>) -> tensor<2x3x1x!FHE.eint<7>>
  return %1: tensor<2x3x1x!FHE.eint<7>>
}
)";
  using tensor3_in = std::array<std::array<std::array<uint8_t, 1>, 3>, 2>;
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<tensor3_out, tensor3_in, tensor3_in>>(outputLib);
  tensor3_in ta = {{{{{1}, {2}, {3}}}, {{{4}, {5}, {6}}}}};
  auto res = lambda.call(ta, ta);
  ASSERT_TRUE(res);
  tensor3_out v = res.value();
  for (size_t i = 0; i < v.size(); i++) {
    for (size_t j = 0; j < v[i].size(); j++) {
      for (size_t k = 0; k < v[i][j].size(); k++) {
        EXPECT_EQ(v[i][j][k], 2 * ta[i][j][k]);
      }
    }
  }
}

static std::string fileContent(std::string path) {
  std::ifstream file(path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

TEST(CompiledModule, call_2t_1s_with_header) {
  std::string source = R"(
func.func @extract(%arg0: tensor<3x!FHE.eint<7>>, %arg1: tensor<3x!FHE.eint<7>>) -> !FHE.eint<7> {
  %1 = "FHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x!FHE.eint<7>>, tensor<3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>
  %c1 = arith.constant 1 : i8
  %2 = tensor.from_elements %c1, %c1, %c1 : tensor<3xi8>
  %3 = "FHELinalg.dot_eint_int"(%1, %2) : (tensor<3x!FHE.eint<7>>, tensor<3xi8>) -> !FHE.eint<7>
  return %3: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  namespace extract = fhecircuit::client::extract;
  auto compiled = compile(outputLib, source, extract::name);
  std::string jsonPath = compiled.getClientParametersPath(outputLib);
  auto cLambda_ = extract::load(jsonPath);
  ASSERT_TRUE(cLambda_);
  tensor1_in ta{1, 2, 3};
  tensor1_in tb{5, 7, 9};
  auto sLambda_ = ServerLambda::load(extract::name, outputLib);
  ASSERT_TRUE(sLambda_);
  auto cLambda = cLambda_.value();
  auto sLambda = sLambda_.value();
  auto keySet_ = cLambda.keySet(getTestKeySetCachePtr(), 0, 0);
  ASSERT_TRUE(keySet_.has_value());
  std::shared_ptr<KeySet> keySet = std::move(keySet_.value());
  auto testLambda = TestTypedLambdaFrom(cLambda, sLambda, keySet);
  auto res = testLambda.call(ta, tb);
  auto expected = std::accumulate(ta.begin(), ta.end(), 0u) +
                  std::accumulate(tb.begin(), tb.end(), 0u);
  ASSERT_EQ_OUTCOME(res, expected);

  EXPECT_EQ(fileContent(THIS_TEST_DIRECTORY +
                        "/call_2t_1s_with_header-client.h.generated"),
            fileContent(OUT_DIRECTORY +
                        "/call_2t_1s_with_header/fhecircuit-client.h"));
}

TEST(DISABLED_CompiledModule, call_2s_1s_lookup_table) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<6>, %arg1: !FHE.eint<3>) -> !FHE.eint<6> {
    %tlu_7 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
    %tlu_3 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %a = "FHE.apply_lookup_table"(%arg0, %tlu_7): (!FHE.eint<6>, tensor<64xi64>) -> (!FHE.eint<6>)
    %b = "FHE.apply_lookup_table"(%arg1, %tlu_3): (!FHE.eint<3>, tensor<8xi64>) -> (!FHE.eint<6>)
    %a_plus_b = "FHE.add_eint"(%a, %b): (!FHE.eint<6>, !FHE.eint<6>) -> (!FHE.eint<6>)
    return %a_plus_b: !FHE.eint<6>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto lambda =
      load<TestTypedLambda<scalar_out, scalar_in, scalar_in>>(outputLib);
  for (auto a : values_6bits())
    for (auto b : values_3bits()) {
      auto res = lambda.call(a, b);
      ASSERT_EQ_OUTCOME(res, a + b);
    }
}
