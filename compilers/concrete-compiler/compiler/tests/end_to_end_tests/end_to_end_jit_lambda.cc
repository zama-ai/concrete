#include <gtest/gtest.h>

#include "concretelang/TestLib/TestProgram.h"
#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"
#include "tests_tools/assert.h"

TEST(Lambda_check_param, int_to_void_missing_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(%arg0: !FHE.eint<1>) {
      return
    }
    )XXX");
  ASSERT_OUTCOME_HAS_FAILURE(lambda.call({}));
}

TEST(Lambda_check_param, DISABLED_int_to_void_good) {
  // DISABLED Note: it segfaults
  checkedJit(lambda, R"XXX(
    func.func @main(%arg0: !FHE.eint<1>) {
      return
    }
    )XXX");
  ASSERT_OUTCOME_HAS_VALUE(lambda.call({Tensor<uint64_t>(1)}));
}

TEST(Lambda_check_param, int_to_void_superfluous_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(%arg0: !FHE.eint<1>) {
      return
    }
    )XXX");
  ASSERT_OUTCOME_HAS_FAILURE(
      lambda.call({Tensor<uint64_t>(1), Tensor<uint64_t>(1)}));
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
  ASSERT_OUTCOME_HAS_FAILURE(lambda.call({}));
  ASSERT_OUTCOME_HAS_FAILURE(lambda.call({Tensor<uint64_t>(1)}));
  ASSERT_OUTCOME_HAS_FAILURE(
      lambda.call({Tensor<uint64_t>(1), Tensor<uint64_t>(2)}));
  ASSERT_OUTCOME_HAS_VALUE(lambda.call(
      {Tensor<uint64_t>(1), Tensor<uint64_t>(2), Tensor<uint64_t>(3)}));
  ASSERT_OUTCOME_HAS_FAILURE(
      lambda.call({Tensor<uint64_t>(1), Tensor<uint64_t>(2),
                   Tensor<uint64_t>(3), Tensor<uint64_t>(4)}));
}

TEST(Lambda_check_param, scalar_tensor_to_scalar_missing_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(
      %arg0: !FHE.eint<1>, %arg1: tensor<2x!FHE.eint<1>>) -> !FHE.eint<1>
    {
      return %arg0: !FHE.eint<1>
    }
  )XXX");
  ASSERT_OUTCOME_HAS_FAILURE(lambda.call({Tensor<uint64_t>(1)}));
}

TEST(Lambda_check_param, scalar_tensor_to_scalar) {
  checkedJit(lambda, R"XXX(
    func.func @main(
      %arg0: !FHE.eint<1>, %arg1: tensor<2x!FHE.eint<1>>) -> !FHE.eint<1>
    {
      return %arg0: !FHE.eint<1>
    }
  )XXX");
  ASSERT_OUTCOME_HAS_VALUE(
      lambda.call({Tensor<uint64_t>(1), Tensor<uint64_t>({1, 2}, {2})}));
}

TEST(Lambda_check_param, scalar_tensor_to_scalar_superfluous_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(
      %arg0: !FHE.eint<1>, %arg1: tensor<2x!FHE.eint<1>>) -> !FHE.eint<1>
    {
      return %arg0: !FHE.eint<1>
    }
  )XXX");
  ASSERT_OUTCOME_HAS_FAILURE(
      lambda.call({Tensor<uint64_t>(1), Tensor<uint64_t>({1, 2}, {2}),
                   Tensor<uint64_t>({1, 2}, {2})}));
}

TEST(Lambda_check_param, scalar_tensor_to_tensor_good_number_param) {
  checkedJit(lambda, R"XXX(
    func.func @main(
      %arg0: !FHE.eint<1>, %arg1: tensor<2x!FHE.eint<1>>) -> tensor<2x!FHE.eint<1>>
    {
      return %arg1: tensor<2x!FHE.eint<1>>
    }
    )XXX");
  ASSERT_OUTCOME_HAS_VALUE(
      lambda.call({Tensor<uint64_t>(1), Tensor<uint64_t>({1, 2}, {2})}));
}

TEST(Lambda_check_param, DISABLED_check_parameters_scalar_too_big) {
  // DISABLED Note: loss of precision without any warning or error.
  checkedJit(lambda, R"XXX(
  func.func @main(%arg0: !FHE.eint<1>) -> !FHE.eint<1>
  {
    return %arg0: !FHE.eint<1>
  }
  )XXX");
  ASSERT_OUTCOME_HAS_FAILURE(lambda.call({Tensor<uint64_t>(3)}));
}
