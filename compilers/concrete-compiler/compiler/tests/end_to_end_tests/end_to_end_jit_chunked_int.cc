#include <gtest/gtest.h>

#include "concretelang/TestLib/TestProgram.h"
#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"

TEST(Lambda_chunked_int, chunked_int_add_eint) {
  checkedJit(testCircuit, R"XXX(
    func.func @main(%arg0: !FHE.eint<64>, %arg1: !FHE.eint<64>) -> !FHE.eint<64> {
      %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<64>, !FHE.eint<64>) -> (!FHE.eint<64>)
      return %1: !FHE.eint<64>
    }
    )XXX",
             "main", DEFAULT_useDefaultFHEConstraints,
             DEFAULT_dataflowParallelize, DEFAULT_loopParallelize,
             DEFAULT_batchTFHEOps, DEFAULT_global_p_error, true, 4, 2);
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };
  ASSERT_EQ(lambda({Tensor<uint64_t>(1), Tensor<uint64_t>(2)}), (uint64_t)3);
  ASSERT_EQ(
      lambda({Tensor<uint64_t>(72057594037927936), Tensor<uint64_t>(10000)}),
      (uint64_t)72057594037937936);
  ASSERT_EQ(
      lambda({Tensor<uint64_t>(2057594037927936), Tensor<uint64_t>(1111)}),
      (uint64_t)2057594037929047);
}
