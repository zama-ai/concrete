
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "end_to_end_jit_test.h"

const mlir::zamalang::V0FHEConstraint defaultV0Constraints{10, 7};

TEST(CompileAndRunHLFHE, add_eint) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !HLFHE.eint<7>, %arg1: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2_u64), 3);
  ASSERT_EXPECTED_VALUE(lambda(4_u64, 5_u64), 9);
  ASSERT_EXPECTED_VALUE(lambda(1_u64, 1_u64), 2);
}

// Same as CompileAndRunHLFHE::add_eint above, but using
// `LambdaArgument` instances as arguments
TEST(CompileAndRunHLFHE, add_eint_lambda_argument) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !HLFHE.eint<7>, %arg1: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
)XXX");

  mlir::zamalang::IntLambdaArgument<> ila1(1);
  mlir::zamalang::IntLambdaArgument<> ila2(2);
  mlir::zamalang::IntLambdaArgument<> ila7(7);
  mlir::zamalang::IntLambdaArgument<> ila9(9);

  ASSERT_EXPECTED_VALUE(lambda({&ila1, &ila2}), 3);
  ASSERT_EXPECTED_VALUE(lambda({&ila7, &ila9}), 16);
  ASSERT_EXPECTED_VALUE(lambda({&ila1, &ila7}), 8);
  ASSERT_EXPECTED_VALUE(lambda({&ila1, &ila9}), 10);
  ASSERT_EXPECTED_VALUE(lambda({&ila2, &ila7}), 9);
}

// Same as CompileAndRunHLFHE::add_eint above, but using
// `LambdaArgument` instances as arguments and as a result type
TEST(CompileAndRunHLFHE, add_eint_lambda_argument_res) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !HLFHE.eint<7>, %arg1: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
)XXX");

  mlir::zamalang::IntLambdaArgument<> ila1(1);
  mlir::zamalang::IntLambdaArgument<> ila2(2);
  mlir::zamalang::IntLambdaArgument<> ila7(7);
  mlir::zamalang::IntLambdaArgument<> ila9(9);

  auto eval = [&](mlir::zamalang::IntLambdaArgument<> &arg0,
                  mlir::zamalang::IntLambdaArgument<> &arg1,
                  uint64_t expected) {
    llvm::Expected<std::unique_ptr<mlir::zamalang::LambdaArgument>> res0 =
        lambda.operator()<std::unique_ptr<mlir::zamalang::LambdaArgument>>(
            {&arg0, &arg1});

    ASSERT_EXPECTED_SUCCESS(res0);
    ASSERT_TRUE((*res0)->isa<mlir::zamalang::IntLambdaArgument<>>());
    ASSERT_EQ((*res0)->cast<mlir::zamalang::IntLambdaArgument<>>().getValue(),
              expected);
  };

  eval(ila1, ila2, 3);
  eval(ila7, ila9, 16);
  eval(ila1, ila7, 8);
  eval(ila1, ila9, 10);
  eval(ila2, ila7, 9);
}

TEST(CompileAndRunHLFHE, add_u64) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
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

TEST(CompileAndRunTensorStd, extract_64) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10xi64>, %i: index) -> i64{
  %c = tensor.extract %t[%i] : tensor<10xi64>
  return %c : i64
}
)XXX",
                                                                "main", "true");

  static uint64_t t_arg[] = {0xFFFFFFFFFFFFFFFF,
                             0,
                             8978,
                             2587490,
                             90,
                             197864,
                             698735,
                             72132,
                             87474,
                             42};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i), t_arg[i]);
}

TEST(CompileAndRunTensorStd, extract_32) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10xi32>, %i: index) -> i32{
  %c = tensor.extract %t[%i] : tensor<10xi32>
  return %c : i32
}
)XXX",
                                                                "main", "true");
  static uint32_t t_arg[] = {0xFFFFFFFF, 0,      8978,  2587490, 90,
                             197864,     698735, 72132, 87474,   42};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i), t_arg[i]);
}

// Same as `CompileAndRunTensorStd::extract_32` above, but using
// `LambdaArgument` instances as arguments
TEST(CompileAndRunTensorStd, extract_32_lambda_argument) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10xi32>, %i: index) -> i32{
  %c = tensor.extract %t[%i] : tensor<10xi32>
  return %c : i32
}
)XXX",
                                                                "main", "true");
  static std::vector<uint32_t> t_arg{0xFFFFFFFF, 0,      8978,  2587490, 90,
                                     197864,     698735, 72132, 87474,   42};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint32_t>>
      tla(t_arg);

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++) {
    mlir::zamalang::IntLambdaArgument<size_t> idx(i);
    ASSERT_EXPECTED_VALUE(lambda({&tla, &idx}), t_arg[i]);
  }
}

TEST(CompileAndRunTensorStd, extract_16) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10xi16>, %i: index) -> i16{
  %c = tensor.extract %t[%i] : tensor<10xi16>
  return %c : i16
}
)XXX",
                                                                "main", "true");

  uint16_t t_arg[] = {0xFFFF, 0,     59589, 47826, 16227,
                      63269,  36435, 52380, 7401,  13313};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i), t_arg[i]);
}

TEST(CompileAndRunTensorStd, extract_8) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10xi8>, %i: index) -> i8{
  %c = tensor.extract %t[%i] : tensor<10xi8>
  return %c : i8
}
)XXX",
                                                                "main", "true");

  static uint8_t t_arg[] = {0xFF, 0, 120, 225, 14, 177, 131, 84, 174, 93};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i), t_arg[i]);
}

TEST(CompileAndRunTensorStd, extract_5) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10xi5>, %i: index) -> i5{
  %c = tensor.extract %t[%i] : tensor<10xi5>
  return %c : i5
}
)XXX",
                                                                "main", "true");

  static uint8_t t_arg[] = {32, 0, 10, 25, 14, 25, 18, 28, 14, 7};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i), t_arg[i]);
}

TEST(CompileAndRunTensorStd, extract_1) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10xi1>, %i: index) -> i1{
  %c = tensor.extract %t[%i] : tensor<10xi1>
  return %c : i1
}
)XXX",
                                                                "main", "true");

  static uint8_t t_arg[] = {0, 0, 1, 0, 1, 1, 0, 1, 1, 0};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i), t_arg[i]);
}

TEST(CompileAndRunTensorEncrypted, extract_5) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10x!HLFHE.eint<5>>, %i: index) -> !HLFHE.eint<5>{
  %c = tensor.extract %t[%i] : tensor<10x!HLFHE.eint<5>>
  return %c : !HLFHE.eint<5>
}
)XXX");

  static uint8_t t_arg[] = {32, 0, 10, 25, 14, 25, 18, 28, 14, 7};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i), t_arg[i]);
}

TEST(CompileAndRunTensorEncrypted, extract_twice_and_add_5) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10x!HLFHE.eint<5>>, %i: index, %j: index) ->
!HLFHE.eint<5>{
  %ti = tensor.extract %t[%i] : tensor<10x!HLFHE.eint<5>>
  %tj = tensor.extract %t[%j] : tensor<10x!HLFHE.eint<5>>
  %c = "HLFHE.add_eint"(%ti, %tj) : (!HLFHE.eint<5>, !HLFHE.eint<5>) ->
  !HLFHE.eint<5> return %c : !HLFHE.eint<5>
}
)XXX");

  static uint8_t t_arg[] = {3, 0, 7, 12, 14, 6, 5, 4, 1, 2};

  for (size_t i = 0; i < ARRAY_SIZE(t_arg); i++)
    for (size_t j = 0; j < ARRAY_SIZE(t_arg); j++)
      ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg), i, j),
                            t_arg[i] + t_arg[j]);
}

TEST(CompileAndRunTensorEncrypted, dim_5) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<10x!HLFHE.eint<5>>) -> index{
  %c0 = arith.constant 0 : index
  %c = tensor.dim %t, %c0 : tensor<10x!HLFHE.eint<5>>
  return %c : index
}
)XXX");

  static uint8_t t_arg[] = {32, 0, 10, 25, 14, 25, 18, 28, 14, 7};
  ASSERT_EXPECTED_VALUE(lambda(t_arg, ARRAY_SIZE(t_arg)), ARRAY_SIZE(t_arg));
}

TEST(CompileAndRunTensorEncrypted, from_elements_5) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%0: !HLFHE.eint<5>) -> tensor<1x!HLFHE.eint<5>> {
  %t = tensor.from_elements %0 : tensor<1x!HLFHE.eint<5>>
  return %t: tensor<1x!HLFHE.eint<5>>
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
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%0: !HLFHE.eint<5>) -> tensor<1x!HLFHE.eint<5>> {
  %t = tensor.from_elements %0 : tensor<1x!HLFHE.eint<5>>
  return %t: tensor<1x!HLFHE.eint<5>>
}
)XXX");

  mlir::zamalang::IntLambdaArgument<> arg(10);

  llvm::Expected<std::unique_ptr<mlir::zamalang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::zamalang::LambdaArgument>>(
          {&arg});

  ASSERT_EXPECTED_SUCCESS(res);
  ASSERT_TRUE((*res)
                  ->isa<mlir::zamalang::TensorLambdaArgument<
                      mlir::zamalang::IntLambdaArgument<>>>());

  mlir::zamalang::TensorLambdaArgument<mlir::zamalang::IntLambdaArgument<>>
      &resp = (*res)
                  ->cast<mlir::zamalang::TensorLambdaArgument<
                      mlir::zamalang::IntLambdaArgument<>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)1);
  ASSERT_EQ(resp.getDimensions().at(0), 1);
  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 1);
  ASSERT_EQ(resp.getValue()[0], 10_u64);
}

TEST(CompileAndRunTensorEncrypted, in_out_tensor_with_op_5) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%in: tensor<2x!HLFHE.eint<5>>) -> tensor<3x!HLFHE.eint<5>> {
  %c_0 = arith.constant 0 : index
  %c_1 = arith.constant 1 : index
  %a = tensor.extract %in[%c_0] : tensor<2x!HLFHE.eint<5>>
  %b = tensor.extract %in[%c_1] : tensor<2x!HLFHE.eint<5>>
  %aplusa = "HLFHE.add_eint"(%a, %a): (!HLFHE.eint<5>, !HLFHE.eint<5>) ->
  (!HLFHE.eint<5>) %aplusb = "HLFHE.add_eint"(%a, %b): (!HLFHE.eint<5>,
  !HLFHE.eint<5>) -> (!HLFHE.eint<5>) %bplusb = "HLFHE.add_eint"(%b, %b):
  (!HLFHE.eint<5>, !HLFHE.eint<5>) -> (!HLFHE.eint<5>) %out =
  tensor.from_elements %aplusa, %aplusb, %bplusb : tensor<3x!HLFHE.eint<5>>
  return %out: tensor<3x!HLFHE.eint<5>>
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

TEST(CompileAndRunTensorEncrypted, linalg_generic) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
func @main(%arg0: tensor<2x!HLFHE.eint<7>>, %arg1: tensor<2xi8>, %acc:
!HLFHE.eint<7>) -> !HLFHE.eint<7> {
  %tacc = tensor.from_elements %acc : tensor<1x!HLFHE.eint<7>>
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types
  = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!HLFHE.eint<7>>, tensor<2xi8>)
  outs(%tacc : tensor<1x!HLFHE.eint<7>>) { ^bb0(%arg2: !HLFHE.eint<7>, %arg3:
  i8, %arg4: !HLFHE.eint<7>):  // no predecessors
    %4 = "HLFHE.mul_eint_int"(%arg2, %arg3) : (!HLFHE.eint<7>, i8) ->
    !HLFHE.eint<7> %5 = "HLFHE.add_eint"(%4, %arg4) : (!HLFHE.eint<7>,
    !HLFHE.eint<7>) -> !HLFHE.eint<7> linalg.yield %5 : !HLFHE.eint<7>
  } -> tensor<1x!HLFHE.eint<7>>
  %c0 = arith.constant 0 : index
  %ret = tensor.extract %2[%c0] : tensor<1x!HLFHE.eint<7>>
  return %ret : !HLFHE.eint<7>
}
)XXX",
                                                                "main", "true");

  static uint8_t arg0[] = {2, 8};
  static uint8_t arg1[] = {6, 8};

  llvm::Expected<uint64_t> res =
      lambda(arg0, ARRAY_SIZE(arg0), arg1, ARRAY_SIZE(arg1), 0_u64);

  ASSERT_EXPECTED_VALUE(res, 76);
}

class CompileAndRunWithPrecision : public ::testing::TestWithParam<int> {};

TEST_P(CompileAndRunWithPrecision, identity_func) {
  uint64_t precision = GetParam();
  std::ostringstream mlirProgram;
  uint64_t sizeOfTLU = 1 << precision;

  mlirProgram << "func @main(%arg0: !HLFHE.eint<" << precision
              << ">) -> !HLFHE.eint<" << precision << "> { \n"
              << "    %tlu = arith.constant dense<[0";

  for (uint64_t i = 1; i < sizeOfTLU; i++)
    mlirProgram << ", " << i;

  mlirProgram << "]> : tensor<" << sizeOfTLU << "xi64>\n"
              << "    %1 = \"HLFHE.apply_lookup_table\"(%arg0, %tlu): "
              << "(!HLFHE.eint<" << precision << ">, tensor<" << sizeOfTLU
              << "xi64>) -> (!HLFHE.eint<" << precision << ">)\n "
              << "return %1: !HLFHE.eint<" << precision << ">\n"
              << "}\n";

  mlir::zamalang::JitCompilerEngine::Lambda lambda =
      checkedJit(mlirProgram.str());

  if (precision == 7) {
    // Test fails with a probability of 5% for a precision of 7. The
    // probability of the test failing 5 times in a row is .05^5,
    // which is less than 1:10,000 and comparable to the probability
    // of failure for the other values.
    static const int max_tries = 3;

    for (uint64_t i = 0; i < sizeOfTLU; i++) {
      for (int retry = 0; retry <= max_tries; retry++) {
        if (retry == max_tries)
          GTEST_FATAL_FAILURE_("Maximum number of tries exceeded");

        llvm::Expected<uint64_t> val = lambda(i);
        ASSERT_EXPECTED_SUCCESS(val);

        if (*val == i)
          break;
      }
    }
  } else {
    for (uint64_t i = 0; i < sizeOfTLU; i++)
      ASSERT_EXPECTED_VALUE(lambda(i), i);
  }
}

INSTANTIATE_TEST_SUITE_P(TestHLFHEApplyLookupTable, CompileAndRunWithPrecision,
                         ::testing::Values(1, 2, 3, 4, 5, 6, 7));

TEST(TestHLFHEApplyLookupTable, multiple_precision) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !HLFHE.eint<6>, %arg1: !HLFHE.eint<3>) -> !HLFHE.eint<6> {
    %tlu_7 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
    %tlu_3 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %a = "HLFHE.apply_lookup_table"(%arg0, %tlu_7): (!HLFHE.eint<6>, tensor<64xi64>) -> (!HLFHE.eint<6>)
    %b = "HLFHE.apply_lookup_table"(%arg1, %tlu_3): (!HLFHE.eint<3>, tensor<8xi64>) -> (!HLFHE.eint<6>)
    %a_plus_b = "HLFHE.add_eint"(%a, %b): (!HLFHE.eint<6>, !HLFHE.eint<6>) -> (!HLFHE.eint<6>)
    return %a_plus_b: !HLFHE.eint<6>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(23_u64, 7_u64), 30);
}

TEST(CompileAndRunTLU, random_func) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !HLFHE.eint<6>) -> !HLFHE.eint<6> {
    %tlu = arith.constant dense<[16, 91, 16, 83, 80, 74, 21, 96, 1, 63, 49, 122, 76, 89, 74, 55, 109, 110, 103, 54, 105, 14, 66, 47, 52, 89, 7, 10, 73, 44, 119, 92, 25, 104, 123, 100, 108, 86, 29, 121, 118, 52, 107, 48, 34, 37, 13, 122, 107, 48, 74, 59, 96, 36, 50, 55, 120, 72, 27, 45, 12, 5, 96, 12]> : tensor<64xi64>
    %1 = "HLFHE.apply_lookup_table"(%arg0, %tlu): (!HLFHE.eint<6>, tensor<64xi64>) -> (!HLFHE.eint<6>)
    return %1: !HLFHE.eint<6>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(5_u64), 74);
  ASSERT_EXPECTED_VALUE(lambda(62_u64), 96);
  ASSERT_EXPECTED_VALUE(lambda(0_u64), 16);
  ASSERT_EXPECTED_VALUE(lambda(63_u64), 12);
}
