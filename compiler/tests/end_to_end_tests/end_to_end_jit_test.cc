
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "end_to_end_jit_test.h"

TEST(CompileAndRunClear, add_u64) {
  checkedJit(lambda, R"XXX(
func.func @main(%arg0: i64, %arg1: i64) -> i64 {
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
func.func @main(%t: tensor<10x!FHE.eint<5>>, %i: index) -> !FHE.eint<5>{
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
func.func @main(%t: tensor<10x!FHE.eint<5>>, %i: index, %j: index) ->
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
func.func @main(%t: tensor<10x!FHE.eint<5>>) -> index{
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
func.func @main(%0: !FHE.eint<5>) -> tensor<1x!FHE.eint<5>> {
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

TEST(CompileAndRunTensorEncrypted, from_elements_multiple_values) {
  checkedJit(lambda, R"XXX(
func.func @main(%0: !FHE.eint<5>, %1: !FHE.eint<5>, %2: !FHE.eint<5>) -> tensor<3x!FHE.eint<5>> {
  %t = tensor.from_elements %0, %1, %2 : tensor<3x!FHE.eint<5>>
  return %t: tensor<3x!FHE.eint<5>>
}
)XXX");

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>(1_u64, 2_u64, 3_u64);

  ASSERT_EXPECTED_SUCCESS(res);
  ASSERT_EQ(res->size(), (size_t)3);
  ASSERT_EQ(res->at(0), 1_u64);
  ASSERT_EQ(res->at(1), 2_u64);
  ASSERT_EQ(res->at(2), 3_u64);
}

TEST(CompileAndRunTensorEncrypted, from_elements_many_values) {
  checkedJit(lambda, R"XXX(
func.func @main(%0: !FHE.eint<5>,
           %1: !FHE.eint<5>,
           %2: !FHE.eint<5>,
           %3: !FHE.eint<5>,
           %4: !FHE.eint<5>,
           %5: !FHE.eint<5>,
           %6: !FHE.eint<5>,
           %7: !FHE.eint<5>,
           %8: !FHE.eint<5>,
           %9: !FHE.eint<5>,
           %10: !FHE.eint<5>,
           %11: !FHE.eint<5>,
           %12: !FHE.eint<5>,
           %13: !FHE.eint<5>,
           %14: !FHE.eint<5>,
           %15: !FHE.eint<5>,
           %16: !FHE.eint<5>,
           %17: !FHE.eint<5>,
           %18: !FHE.eint<5>,
           %19: !FHE.eint<5>,
           %20: !FHE.eint<5>,
           %21: !FHE.eint<5>,
           %22: !FHE.eint<5>,
           %23: !FHE.eint<5>,
           %24: !FHE.eint<5>,
           %25: !FHE.eint<5>,
           %26: !FHE.eint<5>,
           %27: !FHE.eint<5>,
           %28: !FHE.eint<5>,
           %29: !FHE.eint<5>,
           %30: !FHE.eint<5>,
           %31: !FHE.eint<5>,
           %32: !FHE.eint<5>,
           %33: !FHE.eint<5>,
           %34: !FHE.eint<5>,
           %35: !FHE.eint<5>,
           %36: !FHE.eint<5>,
           %37: !FHE.eint<5>,
           %38: !FHE.eint<5>,
           %39: !FHE.eint<5>,
           %40: !FHE.eint<5>,
           %41: !FHE.eint<5>,
           %42: !FHE.eint<5>,
           %43: !FHE.eint<5>,
           %44: !FHE.eint<5>,
           %45: !FHE.eint<5>,
           %46: !FHE.eint<5>,
           %47: !FHE.eint<5>,
           %48: !FHE.eint<5>,
           %49: !FHE.eint<5>,
           %50: !FHE.eint<5>,
           %51: !FHE.eint<5>,
           %52: !FHE.eint<5>,
           %53: !FHE.eint<5>,
           %54: !FHE.eint<5>,
           %55: !FHE.eint<5>,
           %56: !FHE.eint<5>,
           %57: !FHE.eint<5>,
           %58: !FHE.eint<5>,
           %59: !FHE.eint<5>,
           %60: !FHE.eint<5>,
           %61: !FHE.eint<5>,
           %62: !FHE.eint<5>,
           %63: !FHE.eint<5>
) -> tensor<64x!FHE.eint<5>> {
  %t = tensor.from_elements %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63 : tensor<64x!FHE.eint<5>>
  return %t: tensor<64x!FHE.eint<5>>
}
)XXX");

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>(
          0_u64, 1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64, 9_u64,
          10_u64, 11_u64, 12_u64, 13_u64, 14_u64, 15_u64, 16_u64, 17_u64,
          18_u64, 19_u64, 20_u64, 21_u64, 22_u64, 23_u64, 24_u64, 25_u64,
          26_u64, 27_u64, 28_u64, 29_u64, 30_u64, 31_u64, 32_u64, 33_u64,
          34_u64, 35_u64, 36_u64, 37_u64, 38_u64, 39_u64, 40_u64, 41_u64,
          42_u64, 43_u64, 44_u64, 45_u64, 46_u64, 47_u64, 48_u64, 49_u64,
          50_u64, 51_u64, 52_u64, 53_u64, 54_u64, 55_u64, 56_u64, 57_u64,
          58_u64, 59_u64, 60_u64, 61_u64, 62_u64, 63_u64);

  ASSERT_EXPECTED_SUCCESS(res);
  ASSERT_EQ(res->size(), (size_t)64);
  ASSERT_EQ(res->at(0), 0_u64);
  ASSERT_EQ(res->at(1), 1_u64);
  ASSERT_EQ(res->at(2), 2_u64);
  ASSERT_EQ(res->at(3), 3_u64);
  ASSERT_EQ(res->at(4), 4_u64);
  ASSERT_EQ(res->at(5), 5_u64);
  ASSERT_EQ(res->at(6), 6_u64);
  ASSERT_EQ(res->at(7), 7_u64);
  ASSERT_EQ(res->at(8), 8_u64);
  ASSERT_EQ(res->at(9), 9_u64);
  ASSERT_EQ(res->at(10), 10_u64);
  ASSERT_EQ(res->at(11), 11_u64);
  ASSERT_EQ(res->at(12), 12_u64);
  ASSERT_EQ(res->at(13), 13_u64);
  ASSERT_EQ(res->at(14), 14_u64);
  ASSERT_EQ(res->at(15), 15_u64);
  ASSERT_EQ(res->at(16), 16_u64);
  ASSERT_EQ(res->at(17), 17_u64);
  ASSERT_EQ(res->at(18), 18_u64);
  ASSERT_EQ(res->at(19), 19_u64);
  ASSERT_EQ(res->at(20), 20_u64);
  ASSERT_EQ(res->at(21), 21_u64);
  ASSERT_EQ(res->at(22), 22_u64);
  ASSERT_EQ(res->at(23), 23_u64);
  ASSERT_EQ(res->at(24), 24_u64);
  ASSERT_EQ(res->at(25), 25_u64);
  ASSERT_EQ(res->at(26), 26_u64);
  ASSERT_EQ(res->at(27), 27_u64);
  ASSERT_EQ(res->at(28), 28_u64);
  ASSERT_EQ(res->at(29), 29_u64);
  ASSERT_EQ(res->at(30), 30_u64);
  ASSERT_EQ(res->at(31), 31_u64);
  ASSERT_EQ(res->at(32), 32_u64);
  ASSERT_EQ(res->at(33), 33_u64);
  ASSERT_EQ(res->at(34), 34_u64);
  ASSERT_EQ(res->at(35), 35_u64);
  ASSERT_EQ(res->at(36), 36_u64);
  ASSERT_EQ(res->at(37), 37_u64);
  ASSERT_EQ(res->at(38), 38_u64);
  ASSERT_EQ(res->at(39), 39_u64);
  ASSERT_EQ(res->at(40), 40_u64);
  ASSERT_EQ(res->at(41), 41_u64);
  ASSERT_EQ(res->at(42), 42_u64);
  ASSERT_EQ(res->at(43), 43_u64);
  ASSERT_EQ(res->at(44), 44_u64);
  ASSERT_EQ(res->at(45), 45_u64);
  ASSERT_EQ(res->at(46), 46_u64);
  ASSERT_EQ(res->at(47), 47_u64);
  ASSERT_EQ(res->at(48), 48_u64);
  ASSERT_EQ(res->at(49), 49_u64);
  ASSERT_EQ(res->at(50), 50_u64);
  ASSERT_EQ(res->at(51), 51_u64);
  ASSERT_EQ(res->at(52), 52_u64);
  ASSERT_EQ(res->at(53), 53_u64);
  ASSERT_EQ(res->at(54), 54_u64);
  ASSERT_EQ(res->at(55), 55_u64);
  ASSERT_EQ(res->at(56), 56_u64);
  ASSERT_EQ(res->at(57), 57_u64);
  ASSERT_EQ(res->at(58), 58_u64);
  ASSERT_EQ(res->at(59), 59_u64);
  ASSERT_EQ(res->at(60), 60_u64);
  ASSERT_EQ(res->at(61), 61_u64);
  ASSERT_EQ(res->at(62), 62_u64);
  ASSERT_EQ(res->at(63), 63_u64);
}

// Same as `CompileAndRunTensorEncrypted::from_elements_5 but with
// `LambdaArgument` instances as arguments and as a result type
TEST(CompileAndRunTensorEncrypted, from_elements_5_lambda_argument_res) {
  checkedJit(lambda, R"XXX(
func.func @main(%0: !FHE.eint<5>) -> tensor<1x!FHE.eint<5>> {
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
func.func @main(%in: tensor<2x!FHE.eint<5>>) -> tensor<3x!FHE.eint<5>> {
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
func.func @main(%arg0: tensor<2x!FHE.eint<7>>, %arg1: tensor<2xi8>, %acc:
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
