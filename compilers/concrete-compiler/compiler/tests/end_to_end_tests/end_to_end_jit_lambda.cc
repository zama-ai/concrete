#include <gtest/gtest.h>

#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"

TEST(Lambda_check_param, int_to_void_missing_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(%arg0: !FHE.eint<1>) {
      return
    }
    )XXX");
  ASSERT_EXPECTED_FAILURE(lambda());
}

TEST(Lambda_check_param, DISABLED_int_to_void_good) {
  // DISABLED Note: it segfaults
  checkedJit(lambda, R"XXX(
    func.func @main(%arg0: !FHE.eint<1>) {
      return
    }
    )XXX");
  ASSERT_EXPECTED_SUCCESS(lambda(1_u64));
}

TEST(Lambda_check_param, int_to_void_superfluous_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(%arg0: !FHE.eint<1>) {
      return
    }
    )XXX");
  ASSERT_EXPECTED_FAILURE(lambda(1_u64, 1_u64));
}

TEST(Lambda_check_param, scalar_parameters_number) {
  checkedJit(lambda, R"XXX(
  func.func @main(
    %arg0: !FHE.eint<1>, %arg1: !FHE.eint<1>,
    %arg2: !FHE.eint<1>) -> !FHE.eint<1>
  {
    return %arg0: !FHE.eint<1>
  }
  )XXX");
  ASSERT_EXPECTED_FAILURE(lambda());
  ASSERT_EXPECTED_FAILURE(lambda(1_u64));
  ASSERT_EXPECTED_FAILURE(lambda(1_u64, 2_u64));
  ASSERT_EXPECTED_SUCCESS(lambda(1_u64, 2_u64, 3_u64));
  ASSERT_EXPECTED_FAILURE(lambda(1_u64, 2_u64, 3_u64, 4_u64));
}

TEST(Lambda_check_param, scalar_tensor_to_scalar_missing_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(
      %arg0: !FHE.eint<1>, %arg1: tensor<2x!FHE.eint<1>>) -> !FHE.eint<1>
    {
      return %arg0: !FHE.eint<1>
    }
  )XXX");
  ASSERT_EXPECTED_FAILURE(lambda(1_u64));
}

TEST(Lambda_check_param, scalar_tensor_to_scalar) {
  checkedJit(lambda, R"XXX(
    func.func @main(
      %arg0: !FHE.eint<1>, %arg1: tensor<2x!FHE.eint<1>>) -> !FHE.eint<1>
    {
      return %arg0: !FHE.eint<1>
    }
  )XXX");
  uint8_t arg[2] = {1, 2};
  ASSERT_EXPECTED_SUCCESS(lambda(1_u64, arg, ARRAY_SIZE(arg)));
}

TEST(Lambda_check_param, scalar_tensor_to_scalar_superfluous_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(
      %arg0: !FHE.eint<1>, %arg1: tensor<2x!FHE.eint<1>>) -> !FHE.eint<1>
    {
      return %arg0: !FHE.eint<1>
    }
  )XXX");
  uint8_t arg[2] = {1, 2};
  ASSERT_EXPECTED_FAILURE(
      lambda(1_u64, arg, ARRAY_SIZE(arg), arg, ARRAY_SIZE(arg)));
}

TEST(Lambda_check_param, scalar_tensor_to_tensor_good_number_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(
      %arg0: !FHE.eint<1>, %arg1: tensor<2x!FHE.eint<1>>) -> tensor<2x!FHE.eint<1>>
    {
      return %arg1: tensor<2x!FHE.eint<1>>
    }
    )XXX");
  uint8_t arg[2] = {1, 2};
  ASSERT_EXPECTED_SUCCESS(
      lambda.operator()<std::vector<uint8_t>>(1_u64, arg, ARRAY_SIZE(arg)));
}

TEST(Lambda_check_param, DISABLED_check_parameters_scalar_too_big) {
  // DISABLED Note: loss of precision without any warning or error.
  checkedJit(lambda, R"XXX(
  func.func @main(%arg0: !FHE.eint<1>) -> !FHE.eint<1>
  {
    return %arg0: !FHE.eint<1>
  }
  )XXX");
  uint16_t arg = 3;
  ASSERT_EXPECTED_FAILURE(lambda(arg));
}
