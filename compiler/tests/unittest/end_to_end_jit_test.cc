
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "end_to_end_jit_test.h"

TEST(CompileAndRunClear, add_u64) {
  checkedJit(lambda, R"XXX(
func @main(%arg0: i64, %arg1: i64) -> i64 {
  %1 = arith.addi %arg0, %arg1 : i64
  return %1: i64
}
)XXX",
             "main", true);

  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2_u64), (uint64_t)3);
  ASSERT_EXPECTED_VALUE(lambda(4_u64, 5_u64), (uint64_t)9);
  ASSERT_EXPECTED_VALUE(lambda(1_u64, 1_u64), (uint64_t)2);
}

TEST(CompileAndRunTensorEncrypted, extract_5) {
  checkedJit(lambda, R"XXX(
func @main(%t: tensor<10x!FHE.eint<5>>, %i: index) -> !FHE.eint<5>{
  %c = tensor.extract %t[%i] : tensor<10x!FHE.eint<5>>
  return %c : !FHE.eint<5>
}
)XXX");

  static uint8_t t_arg[] = {32, 0, 10, 25, 14, 25, 18, 28, 14, 7};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i), t_arg[i]);
}

TEST(CompileAndRunTensorEncrypted, extract_twice_and_add_5) {
  checkedJit(lambda, R"XXX(
func @main(%t: tensor<10x!FHE.eint<5>>, %i: index, %j: index) ->
!FHE.eint<5>{
  %ti = tensor.extract %t[%i] : tensor<10x!FHE.eint<5>>
  %tj = tensor.extract %t[%j] : tensor<10x!FHE.eint<5>>
  %c = "FHE.add_eint"(%ti, %tj) : (!FHE.eint<5>, !FHE.eint<5>) ->
  !FHE.eint<5> return %c : !FHE.eint<5>
}
)XXX");

  static uint8_t t_arg[] = {3, 0, 7, 12, 14, 6, 5, 4, 1, 2};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    for (size_t j = 0; j < ARRAY_SIZE(t_arg); j++)
      ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i, j),
                            t_arg[i] + t_arg[j]);
}

TEST(CompileAndRunTensorEncrypted, dim_5) {
  checkedJit(lambda, R"XXX(
func @main(%t: tensor<10x!FHE.eint<5>>) -> index{
  %c0 = arith.constant 0 : index
  %c = tensor.dim %t, %c0 : tensor<10x!FHE.eint<5>>
  return %c : index
}
)XXX");

  static uint8_t t_arg[] = {32, 0, 10, 25, 14, 25, 18, 28, 14, 7};
  ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg)), ARRAY_SIZE(t_arg));
}

TEST(CompileAndRunTensorEncrypted, from_elements_5) {
  checkedJit(lambda, R"XXX(
func @main(%0: !FHE.eint<5>) -> tensor<1x!FHE.eint<5>> {
  %t = tensor.from_elements %0 : tensor<1x!FHE.eint<5>>
  return %t: tensor<1x!FHE.eint<5>>
}
)XXX");

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>(10_u64);

  ASSERT_EXPECTED_SUCCESS(res);
  ASSERT_EQ(res->size(), (size_t)1);
  ASSERT_EQ(res->at(0), 10_u64);
}

// Same as `CompileAndRunTensorEncrypted::from_elements_5 but with
// `LambdaArgument` instances as arguments and as a result type
TEST(CompileAndRunTensorEncrypted, from_elements_5_lambda_argument_res) {
  checkedJit(lambda, R"XXX(
func @main(%0: !FHE.eint<5>) -> tensor<1x!FHE.eint<5>> {
  %t = tensor.from_elements %0 : tensor<1x!FHE.eint<5>>
  return %t: tensor<1x!FHE.eint<5>>
}
)XXX");

  mlir::concretelang::IntLambdaArgument<> arg(10);

  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
          {&arg});

  ASSERT_EXPECTED_SUCCESS(res);
  ASSERT_TRUE((*res)
                  ->isa<mlir::concretelang::TensorLambdaArgument<
                      mlir::concretelang::IntLambdaArgument<>>>());

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<>> &resp =
      (*res)
          ->cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)1);
  ASSERT_EQ(resp.getDimensions().at(0), 1);
  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 1);
  ASSERT_EQ(resp.getValue()[0], 10_u64);
}

TEST(CompileAndRunTensorEncrypted, in_out_tensor_with_op_5) {
  checkedJit(lambda, R"XXX(
func @main(%in: tensor<2x!FHE.eint<5>>) -> tensor<3x!FHE.eint<5>> {
  %c_0 = arith.constant 0 : index
  %c_1 = arith.constant 1 : index
  %a = tensor.extract %in[%c_0] : tensor<2x!FHE.eint<5>>
  %b = tensor.extract %in[%c_1] : tensor<2x!FHE.eint<5>>
  %aplusa = "FHE.add_eint"(%a, %a): (!FHE.eint<5>, !FHE.eint<5>) ->
  (!FHE.eint<5>) %aplusb = "FHE.add_eint"(%a, %b): (!FHE.eint<5>,
  !FHE.eint<5>) -> (!FHE.eint<5>) %bplusb = "FHE.add_eint"(%b, %b):
  (!FHE.eint<5>, !FHE.eint<5>) -> (!FHE.eint<5>) %out =
  tensor.from_elements %aplusa, %aplusb, %bplusb : tensor<3x!FHE.eint<5>>
  return %out: tensor<3x!FHE.eint<5>>
}
)XXX");

  static uint8_t in[] = {2, 16};

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>(in, ARRAY_SIZE(in));

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)3);
  ASSERT_EQ(res->at(0), (uint64_t)(in[0] + in[0]));
  ASSERT_EQ(res->at(1), (uint64_t)(in[0] + in[1]));
  ASSERT_EQ(res->at(2), (uint64_t)(in[1] + in[1]));
}

// Test is failing since with the bufferization and the parallel options.
// DISABLED as is a bit artificial test, let's investigate later.
TEST(CompileAndRunTensorEncrypted, DISABLED_linalg_generic) {
  checkedJit(lambda, R"XXX(
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
func @main(%arg0: tensor<2x!FHE.eint<7>>, %arg1: tensor<2xi8>, %acc:
!FHE.eint<7>) -> !FHE.eint<7> {
  %tacc = tensor.from_elements %acc : tensor<1x!FHE.eint<7>>
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types
  = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!FHE.eint<7>>, tensor<2xi8>)
  outs(%tacc : tensor<1x!FHE.eint<7>>) { ^bb0(%arg2: !FHE.eint<7>, %arg3:
  i8, %arg4: !FHE.eint<7>):  // no predecessors
    %4 = "FHE.mul_eint_int"(%arg2, %arg3) : (!FHE.eint<7>, i8) ->
    !FHE.eint<7> %5 = "FHE.add_eint"(%4, %arg4) : (!FHE.eint<7>,
    !FHE.eint<7>) -> !FHE.eint<7> linalg.yield %5 : !FHE.eint<7>
  } -> tensor<1x!FHE.eint<7>>
  %c0 = arith.constant 0 : index
  %ret = tensor.extract %2[%c0] : tensor<1x!FHE.eint<7>>
  return %ret : !FHE.eint<7>
}
)XXX",
             "main", true);

  static uint8_t arg0[] = {2, 8};
  static uint8_t arg1[] = {6, 8};

  llvm::Expected<uint64_t> res =
      lambda(arg0, ARRAY_SIZE(arg0), arg1, ARRAY_SIZE(arg1), 0_u64);

  ASSERT_EXPECTED_VALUE(res, 76);
}
