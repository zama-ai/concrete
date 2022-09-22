
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "end_to_end_gpu_test.h"
#include "tests_tools/GtestEnvironment.h"

TEST(GPULookupTable, lut_precision2) {
  checkedJit(lambda, R"XXX(
func.func @main(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  %arg1 = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi64>
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
)XXX",
             "main", true);

  ASSERT_EXPECTED_VALUE(lambda(0_u64), (uint64_t)1);
  ASSERT_EXPECTED_VALUE(lambda(1_u64), (uint64_t)2);
  ASSERT_EXPECTED_VALUE(lambda(2_u64), (uint64_t)3);
  ASSERT_EXPECTED_VALUE(lambda(3_u64), (uint64_t)0);
}

TEST(GPULookupTable, lut_precision4) {
  checkedJit(lambda, R"XXX(
func.func @main(%arg0: !FHE.eint<4>) -> !FHE.eint<4> {
  %arg1 = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0]> : tensor<16xi64>
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<4>, tensor<16xi64>) -> (!FHE.eint<4>)
  return %1: !FHE.eint<4>
}
)XXX",
             "main", true);

  ASSERT_EXPECTED_VALUE(lambda(0_u64), (uint64_t)1);
  ASSERT_EXPECTED_VALUE(lambda(1_u64), (uint64_t)2);
  ASSERT_EXPECTED_VALUE(lambda(7_u64), (uint64_t)8);
  ASSERT_EXPECTED_VALUE(lambda(15_u64), (uint64_t)0);
}

TEST(GPULookupTable, lut_precision7) {
  checkedJit(lambda, R"XXX(
func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %arg1 = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 0]> : tensor<128xi64>
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)XXX",
             "main", true);

  ASSERT_EXPECTED_VALUE(lambda(0_u64), (uint64_t)1);
  ASSERT_EXPECTED_VALUE(lambda(1_u64), (uint64_t)2);
  ASSERT_EXPECTED_VALUE(lambda(120_u64), (uint64_t)121);
  ASSERT_EXPECTED_VALUE(lambda(127_u64), (uint64_t)0);
}
