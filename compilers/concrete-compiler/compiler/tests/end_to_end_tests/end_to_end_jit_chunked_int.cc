#include <gtest/gtest.h>

#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"

TEST(Lambda_chunked_int, chunked_int_add_eint) {
  checkedJit(lambda, R"XXX(
    func.func @main(%arg0: !FHE.eint<64>, %arg1: !FHE.eint<64>) -> !FHE.eint<64> {
      %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<64>, !FHE.eint<64>) -> (!FHE.eint<64>)
      return %1: !FHE.eint<64>
    }
    )XXX",
             "main", DEFAULT_useDefaultFHEConstraints,
             DEFAULT_dataflowParallelize, DEFAULT_loopParallelize,
             DEFAULT_batchTFHEOps, DEFAULT_global_p_error, true, 4, 2);
  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2_u64), (uint64_t)3);
  ASSERT_EXPECTED_VALUE(lambda(72057594037927936_u64, 10000_u64),
                        (uint64_t)72057594037937936);
  ASSERT_EXPECTED_VALUE(lambda(2057594037927936_u64, 1111_u64),
                        (uint64_t)2057594037929047);
}
