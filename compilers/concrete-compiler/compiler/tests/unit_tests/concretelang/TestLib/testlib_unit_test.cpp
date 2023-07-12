#include <gtest/gtest.h>

#include <cassert>
#include <fstream>
#include <numeric>

#include "boost/outcome.h"

#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/TestLib/TestCircuit.h"

#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"
#include "tests_tools/keySetCache.h"

using concretelang::testlib::TestCircuit;

testing::Environment *const dfr_env =
    testing::AddGlobalTestEnvironment(new DFREnvironment);

const std::string FUNCNAME = "main";

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

TestCircuit load(mlir::concretelang::CompilerEngine::Library compiled) {
  auto keyset = getTestKeySetCachePtr()
                    ->getKeyset(compiled.getProgramInfo().keyset(), 0, 0)
                    .value();
  return TestCircuit::create(keyset, compiled.getProgramInfo(),
                             compiled.getOutputDirPath(), 0, 0, false)
      .value();
}

// TEST(CompiledModule, call_1s_1s_client_view) {
//   std::string source = R"(
// func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
//   return %arg0: !FHE.eint<7>
// }
// )";
//   std::string outputLib = outputLibFromThis(this->test_info_);
//   auto circuit = load(compile(outputLib, source));
//   std::string jsonPath = compiled.getProgramInfoPath(outputLib);
//   auto maybeLambda = MyLambda::load("main", jsonPath);
//   ASSERT_TRUE(maybeLambda.has_value());
//   auto lambda = maybeLambda.value();
//   auto maybeKeySet = lambda.keySet(getTestKeySetCachePtr(), 0, 0);
//   ASSERT_TRUE(maybeKeySet.has_value());
//   std::shared_ptr<KeySet> keySet = std::move(maybeKeySet.value());
//   auto maybePublicArguments = lambda.publicArguments(1, *keySet);

//   ASSERT_TRUE(maybePublicArguments.has_value());
//   auto publicArguments = std::move(maybePublicArguments.value());
//   std::ostringstream osstream(std::ios::binary);
//   ASSERT_TRUE(publicArguments->serialize(osstream).has_value());
//   EXPECT_TRUE(osstream.good());
//   // Direct call without intermediate
//   EXPECT_TRUE(lambda.serializeCall(1, *keySet, osstream));
//   EXPECT_TRUE(osstream.good());
// }

TEST(CompiledModule, call_1s_1s) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  return %arg0: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto compiled = compile(outputLib, source);
  auto circuit = load(compiled);
  for (auto a : values_7bits()) {
    auto res = circuit.call({Tensor<uint64_t>(a)});
    ASSERT_TRUE(res.has_value());
    auto out = res.value()[0].getTensor<uint64_t>().value()[0];
    ASSERT_EQ(out, (uint64_t)a);
  }
}

TEST(CompiledModule, call_2s_1s_choose) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  return %arg0: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto circuit = load(compile(outputLib, source));
  for (auto a : values_7bits())
    for (auto b : values_7bits()) {
      if (a > b) {
        continue;
      }
      auto res = circuit.call({Tensor<uint64_t>(a), Tensor<uint64_t>(b)});
      ASSERT_TRUE(res.has_value());
      auto out = res.value()[0].getTensor<uint64_t>().value()[0];
      ASSERT_EQ(out, (uint64_t)a);
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
  auto circuit = load(compile(outputLib, source));
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

TEST(CompiledModule, call_1s_1s_bad_call) {
  std::string source = R"(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto circuit = load(compile(outputLib, source));
  auto res = circuit.call({Tensor<uint64_t>(1)});
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
  auto circuit = load(compile(outputLib, source));
  for (auto a : values_7bits()) {
    auto res = circuit.call({Tensor<uint64_t>(a)});
    EXPECT_TRUE(res);
    auto out = res.value()[0].getTensor<uint64_t>().value()[0];
    EXPECT_EQ(out, (uint64_t)a);
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
  auto circuit = load(compile(outputLib, source));
  for (auto a : values_7bits()) {
    auto res = circuit.call({Tensor<uint64_t>(a), Tensor<uint64_t>(a + 1)});
    EXPECT_TRUE(res);
    auto out = res.value()[0].getTensor<uint64_t>().value();
    EXPECT_EQ(out[0], (uint64_t)a);
    EXPECT_EQ(out[1], (uint64_t)(a + 1));
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
  auto circuit = load(compile(outputLib, source));
  for (uint8_t a : values_7bits()) {
    auto ta = Tensor<uint64_t>({a}, {1});
    auto res = circuit.call({ta});
    EXPECT_TRUE(res);
    auto out = res.value()[0].getTensor<uint64_t>().value()[0];
    EXPECT_EQ(out, (uint64_t)a);
  }
}

TEST(CompiledModule, call_1t_1t) {
  std::string source = R"(
func.func @main(%arg0: tensor<3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
  return %arg0: tensor<3x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto circuit = load(compile(outputLib, source));
  auto ta = Tensor<uint64_t>({1, 2, 3}, {3});
  auto res = circuit.call({ta});
  ASSERT_TRUE(res);
  auto out = res.value()[0].getTensor<uint64_t>().value();
  EXPECT_EQ(out, ta);
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
  auto circuit = load(compile(outputLib, source));
  auto ta = Tensor<uint64_t>({1, 2, 3}, {3});
  auto tb = Tensor<uint8_t>({5, 7, 9}, {3});
  auto res = circuit.call({ta, tb});
  auto expected = std::accumulate(ta.values.begin(), ta.values.end(), 0u) +
                  std::accumulate(tb.values.begin(), tb.values.end(), 0u);
  ASSERT_TRUE(res);
  auto out = res.value()[0].getTensor<uint64_t>().value()[0];
  ASSERT_EQ(out, expected);
}

TEST(CompiledModule, call_1tr2_1tr2) {
  std::string source = R"(
func.func @main(%arg0: tensor<2x3x!FHE.eint<7>>) -> tensor<2x3x!FHE.eint<7>> {
  return %arg0: tensor<2x3x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto circuit = load(compile(outputLib, source));
  auto ta = Tensor<uint64_t>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto res = circuit.call({ta});
  ASSERT_TRUE(res);
  auto out = res.value()[0].getTensor<uint64_t>().value();
  EXPECT_EQ(out, ta);
}

TEST(CompiledModule, call_1tr3_1tr3) {
  std::string source = R"(
func.func @main(%arg0: tensor<2x3x1x!FHE.eint<7>>) -> tensor<2x3x1x!FHE.eint<7>> {
  return %arg0: tensor<2x3x1x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto circuit = load(compile(outputLib, source));
  auto ta = Tensor<uint64_t>({1, 2, 3, 4, 5, 6}, {2, 3, 1});
  auto res = circuit.call({ta});
  ASSERT_TRUE(res);
  auto out = res.value()[0].getTensor<uint64_t>().value();
  EXPECT_EQ(out, ta);
}

TEST(CompiledModule, call_2tr3_1tr3) {
  std::string source = R"(
func.func @main(%arg0: tensor<2x3x1x!FHE.eint<7>>, %arg1: tensor<2x3x1x!FHE.eint<7>>) -> tensor<2x3x1x!FHE.eint<7>> {
  %1 = "FHELinalg.add_eint"(%arg0, %arg1): (tensor<2x3x1x!FHE.eint<7>>, tensor<2x3x1x!FHE.eint<7>>) -> tensor<2x3x1x!FHE.eint<7>>
  return %1: tensor<2x3x1x!FHE.eint<7>>
}
)";
  std::string outputLib = outputLibFromThis(this->test_info_);
  auto circuit = load(compile(outputLib, source));
  auto ta = Tensor<uint64_t>({1, 2, 3, 4, 5, 6}, {2, 3, 1});
  auto res = circuit.call({ta, ta});
  ASSERT_TRUE(res);
  auto out = res.value()[0].getTensor<uint64_t>().value();
  EXPECT_EQ(out, ta * 2);
}

// static std::string fileContent(std::string path) {
//   std::ifstream file(path);
//   std::stringstream buffer;
//   buffer << file.rdbuf();
//   return buffer.str();
// }

// TEST(CompiledModule, call_2t_1s_with_header) {
//   std::string source = R"(
// func.func @extract(%arg0: tensor<3x!FHE.eint<7>>, %arg1:
// tensor<3x!FHE.eint<7>>) -> !FHE.eint<7> {
//   %1 = "FHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x!FHE.eint<7>>,
//   tensor<3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> %c1 = arith.constant 1 :
//   i8 %2 = tensor.from_elements %c1, %c1, %c1 : tensor<3xi8> %3 =
//   "FHELinalg.dot_eint_int"(%1, %2) : (tensor<3x!FHE.eint<7>>, tensor<3xi8>)
//   -> !FHE.eint<7> return %3: !FHE.eint<7>
// }
// )";
//   std::string outputLib = outputLibFromThis(this->test_info_);
//   namespace extract = fhecircuit::client::extract;
//   auto compiled = load(compile(outputLib, source, extract::name);
//   std::string jsonPath = compiled.getProgramInfoPath(outputLib);
//   auto ccircuit_ = extract::load(jsonPath);
//   ASSERT_TRUE(ccircuit_);
//   tensor1_in ta{1, 2, 3};
//   tensor1_in tb{5, 7, 9};
//   auto scircuit_ = Servercircuit::load(extract::name, outputLib);
//   ASSERT_TRUE(scircuit_);
//   auto ccircuit = ccircuit_.value();
//   auto scircuit = scircuit_.value();
//   auto keySet_ = ccircuit.keySet(getTestKeySetCachePtr(), 0, 0);
//   ASSERT_TRUE(keySet_.has_value());
//   std::shared_ptr<KeySet> keySet = std::move(keySet_.value());
//   auto testcircuit = TestTypedcircuitFrom(ccircuit, scircuit, keySet);
//   auto res = testcircuit.call(ta, tb);
//   auto expected = std::accumulate(ta.begin(), ta.end(), 0u) +
//                   std::accumulate(tb.begin(), tb.end(), 0u);
//   ASSERT_EQ_OUTCOME(res, expected);

//   EXPECT_EQ(fileContent(THIS_TEST_DIRECTORY +
//                         "/call_2t_1s_with_header-client.h.generated"),
//             fileContent(OUT_DIRECTORY +
//                         "/call_2t_1s_with_header/fhecircuit-client.h"));
// }

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
  auto circuit = load(compile(outputLib, source));
  for (auto a : values_6bits())
    for (auto b : values_3bits()) {
      auto res = circuit.call({Tensor<uint64_t>(a), Tensor<uint64_t>(b)});
      ASSERT_TRUE(res);
      auto out = res.value()[0].getTensor<uint64_t>().value()[0];
      ASSERT_EQ(out, (uint64_t)a + b);
    }
}
