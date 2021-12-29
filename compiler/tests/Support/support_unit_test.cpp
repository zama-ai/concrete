#include <gtest/gtest.h>

#include "../unittest/end_to_end_jit_test.h"
#include "concretelang/ClientLib/ClientParameters.h"

namespace CL = mlir::concretelang;

TEST(Support, client_parameters_json_serde) {
  mlir::concretelang::ClientParameters params0;
  params0.secretKeys = {
    {mlir::concretelang::SMALL_KEY, {/*.size = */ 12}},
    {mlir::concretelang::BIG_KEY, {/*.size = */ 14}},
  };
  params0.bootstrapKeys = {
    {
      "bsk_v0", {
        /*.inputSecretKeyID = */ mlir::concretelang::SMALL_KEY,
        /*.outputSecretKeyID = */ mlir::concretelang::BIG_KEY,
        /*.level = */ 1,
        /*.baseLog = */ 2,
        /*.k = */ 3,
        /*.variance = */ 0.001
        }
    },{
      "wtf_bsk_v0", {
        /*.inputSecretKeyID = */ mlir::concretelang::BIG_KEY,
        /*.outputSecretKeyID = */ mlir::concretelang::SMALL_KEY,
        /*.level = */ 3,
        /*.baseLog = */ 2,
        /*.k = */ 1,
        /*.variance = */ 0.0001,
        }
    },
  };
  params0.keyswitchKeys = {
      {
        "ksk_v0", {
          /*.inputSecretKeyID = */ mlir::concretelang::BIG_KEY,
          /*.outputSecretKeyID = */ mlir::concretelang::SMALL_KEY,
          /*.level = */ 1,
          /*.baseLog = */ 2,
          /*.variance = */ 3,
          }
      }
  };
  params0.inputs = {
    {
        /*.encryption = */ {{CL::SMALL_KEY, 0.01, {4}}},
        /*.shape = */ {32, {1, 2, 3, 4}, 1 * 2 * 3 * 4},
    },
    {
        /*.encryption = */ {{CL::SMALL_KEY, 0.03, {5}}},
        /*.shape = */ {8, {4, 4, 4, 4}, 4 * 4 * 4 * 4},
    },
  };
  params0.outputs = {
    {
        /*.encryption = */ {{CL::SMALL_KEY, 0.03, {5}}},
        /*.shape = */ {8, {4, 4, 4, 4}, 4 * 4 * 4 * 4},
    },
   };
  auto json = mlir::concretelang::toJSON(params0);
  std::string jsonStr;
  llvm::raw_string_ostream os(jsonStr);
  os << json;
  auto parseResult =
      llvm::json::parse<mlir::concretelang::ClientParameters>(jsonStr);
  ASSERT_EXPECTED_VALUE(parseResult, params0);
}
