
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "end_to_end_jit_test.h"

///////////////////////////////////////////////////////////////////////////////
// FHE types and operators //////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// FHE.eint /////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHE, identity) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  return %arg0: !FHE.eint<3>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(1_u64), 1);
  ASSERT_EXPECTED_VALUE(lambda(4_u64), 4);
  ASSERT_EXPECTED_VALUE(lambda(8_u64), 8);
}

// FHE.add_eint_int /////////////////////////////////////////////////////////

TEST(End2EndJit_FHE, add_eint_int_cst) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  %0 = arith.constant 1 : i3
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(0_u64), 1);
  ASSERT_EXPECTED_VALUE(lambda(1_u64), 2);
  ASSERT_EXPECTED_VALUE(lambda(2_u64), 3);
  ASSERT_EXPECTED_VALUE(lambda(3_u64), 4);
}

TEST(End2EndJit_FHE, add_eint_int_arg) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<2>, %arg1: i3) -> !FHE.eint<2> {
  %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(0_u64, 1_u64), 1);
  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2_u64), 3);
}

// FHE.sub_int_eint /////////////////////////////////////////////////////////

TEST(End2EndJit_FHE, sub_int_eint_cst) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  %0 = arith.constant 7 : i3
  %1 = "FHE.sub_int_eint"(%0, %arg0): (i3, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(1_u64), 6);
  ASSERT_EXPECTED_VALUE(lambda(2_u64), 5);
  ASSERT_EXPECTED_VALUE(lambda(3_u64), 4);
  ASSERT_EXPECTED_VALUE(lambda(4_u64), 3);
}

TEST(End2EndJit_FHE, sub_int_eint_arg) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: i3, %arg1: !FHE.eint<2>) -> !FHE.eint<2> {
  %1 = "FHE.sub_int_eint"(%arg0, %arg1): (i3, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(2_u64, 2_u64), 0);
  ASSERT_EXPECTED_VALUE(lambda(2_u64, 1_u64), 1);
  ASSERT_EXPECTED_VALUE(lambda(7_u64, 2_u64), 5);
}

// FHE.neg_eint /////////////////////////////////////////////////////////////

TEST(End2EndJit_FHE, neg_eint) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>

}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(0_u64), 0);
  ASSERT_EXPECTED_VALUE(lambda(1_u64), 255);
  ASSERT_EXPECTED_VALUE(lambda(4_u64), 252);
  ASSERT_EXPECTED_VALUE(lambda(250_u64), 6);
}

TEST(End2EndJit_FHE, neg_eint_3bits) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<3>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(0_u64), 0);
  ASSERT_EXPECTED_VALUE(lambda(1_u64), 15);
  ASSERT_EXPECTED_VALUE(lambda(4_u64), 12);
  ASSERT_EXPECTED_VALUE(lambda(13_u64), 3);
}

// FHE.sub_int_eint /////////////////////////////////////////////////////////

TEST(End2EndJit_FHE, mul_eint_int_cst) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  %0 = arith.constant 2 : i3
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(0_u64), 0);
  ASSERT_EXPECTED_VALUE(lambda(1_u64), 2);
  ASSERT_EXPECTED_VALUE(lambda(2_u64), 4);
}

TEST(End2EndJit_FHE, mul_eint_int_arg) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<2>, %arg1: i3) -> !FHE.eint<2> {
  %1 = "FHE.mul_eint_int"(%arg0, %arg1): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(0_u64, 2), 0);
  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2), 2);
  ASSERT_EXPECTED_VALUE(lambda(2_u64, 2), 4);
}

// FHE.add_eint /////////////////////////////////////////////////////////////

TEST(End2EndJit_FHE, add_eint) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2_u64), 3);
  ASSERT_EXPECTED_VALUE(lambda(4_u64, 5_u64), 9);
  ASSERT_EXPECTED_VALUE(lambda(1_u64, 1_u64), 2);
}

// Same as End2EndJit_FHE::add_eint above, but using
// `LambdaArgument` instances as arguments
TEST(End2EndJit_FHE, add_eint_lambda_argument) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)XXX");

  mlir::concretelang::IntLambdaArgument<> ila1(1);
  mlir::concretelang::IntLambdaArgument<> ila2(2);
  mlir::concretelang::IntLambdaArgument<> ila7(7);
  mlir::concretelang::IntLambdaArgument<> ila9(9);

  ASSERT_EXPECTED_VALUE(lambda({&ila1, &ila2}), 3);
  ASSERT_EXPECTED_VALUE(lambda({&ila7, &ila9}), 16);
  ASSERT_EXPECTED_VALUE(lambda({&ila1, &ila7}), 8);
  ASSERT_EXPECTED_VALUE(lambda({&ila1, &ila9}), 10);
  ASSERT_EXPECTED_VALUE(lambda({&ila2, &ila7}), 9);
}

// Same as End2EndJit_FHE::add_eint above, but using
// `LambdaArgument` instances as arguments and as a result type
TEST(End2EndJit_FHE, add_eint_lambda_argument_res) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)XXX");

  mlir::concretelang::IntLambdaArgument<> ila1(1);
  mlir::concretelang::IntLambdaArgument<> ila2(2);
  mlir::concretelang::IntLambdaArgument<> ila7(7);
  mlir::concretelang::IntLambdaArgument<> ila9(9);

  auto eval = [&](mlir::concretelang::IntLambdaArgument<> &arg0,
                  mlir::concretelang::IntLambdaArgument<> &arg1,
                  uint64_t expected) {
    llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> res0 =
        lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
            {&arg0, &arg1});

    ASSERT_EXPECTED_SUCCESS(res0);
    ASSERT_TRUE((*res0)->isa<mlir::concretelang::IntLambdaArgument<>>());
    ASSERT_EQ(
        (*res0)->cast<mlir::concretelang::IntLambdaArgument<>>().getValue(),
        expected);
  };

  eval(ila1, ila2, 3);
  eval(ila7, ila9, 16);
  eval(ila1, ila7, 8);
  eval(ila1, ila9, 10);
  eval(ila2, ila7, 9);
}

// Same as End2EndJit_FHE::neg_eint above, but using
// `LambdaArgument` instances as arguments
TEST(End2EndJit_FHE, neg_eint_lambda_argument) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)XXX");

  mlir::concretelang::IntLambdaArgument<> ila0(0);
  mlir::concretelang::IntLambdaArgument<> ila2(2);
  mlir::concretelang::IntLambdaArgument<> ila7(7);
  mlir::concretelang::IntLambdaArgument<> ila150(150);
  mlir::concretelang::IntLambdaArgument<> ila249(249);

  ASSERT_EXPECTED_VALUE(lambda({&ila0}), 0);
  ASSERT_EXPECTED_VALUE(lambda({&ila2}), 254);
  ASSERT_EXPECTED_VALUE(lambda({&ila7}), 249);
  ASSERT_EXPECTED_VALUE(lambda({&ila150}), 106);
  ASSERT_EXPECTED_VALUE(lambda({&ila249}), 7);
}

// Same as End2EndJit_FHE::neg_eint above, but using
// `LambdaArgument` instances as arguments and as a result type
TEST(End2EndJit_FHE, neg_eint_lambda_argument_res) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
)XXX");

  mlir::concretelang::IntLambdaArgument<> ila1(1);
  mlir::concretelang::IntLambdaArgument<> ila2(2);
  mlir::concretelang::IntLambdaArgument<> ila7(7);
  mlir::concretelang::IntLambdaArgument<> ila9(9);

  auto eval = [&](mlir::concretelang::IntLambdaArgument<> &arg0,
                  uint64_t expected) {
    llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> res0 =
        lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
            {&arg0});

    ASSERT_EXPECTED_SUCCESS(res0);
    ASSERT_TRUE((*res0)->isa<mlir::concretelang::IntLambdaArgument<>>());
    ASSERT_EQ(
        (*res0)->cast<mlir::concretelang::IntLambdaArgument<>>().getValue(),
        expected);
  };

  eval(ila1, 255);
  eval(ila2, 254);
  eval(ila7, 249);
  eval(ila9, 247);
}

// FHE.apply_lookup_table /////////////////////////////////////////////////////

class ApplyLookupTableWithPrecision : public ::testing::TestWithParam<int> {};

TEST_P(ApplyLookupTableWithPrecision, identity_func) {
  uint64_t precision = GetParam();
  std::ostringstream mlirProgram;
  uint64_t sizeOfTLU = 1 << precision;

  mlirProgram << "func @main(%arg0: !FHE.eint<" << precision
              << ">) -> !FHE.eint<" << precision << "> { \n"
              << "    %tlu = arith.constant dense<[0";

  for (uint64_t i = 1; i < sizeOfTLU; i++)
    mlirProgram << ", " << i;

  mlirProgram << "]> : tensor<" << sizeOfTLU << "xi64>\n"
              << "    %1 = \"FHE.apply_lookup_table\"(%arg0, %tlu): "
              << "(!FHE.eint<" << precision << ">, tensor<" << sizeOfTLU
              << "xi64>) -> (!FHE.eint<" << precision << ">)\n "
              << "return %1: !FHE.eint<" << precision << ">\n"
              << "}\n";

  mlir::concretelang::JitCompilerEngine::Lambda lambda =
      checkedJit(mlirProgram.str());

  if (precision >= 6) {
    // This test often fails for this precision, so we need retries.
    // Reason: the current encryption parameters are a little short for this
    // precision.

    static const int max_tries = 10;

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

INSTANTIATE_TEST_SUITE_P(End2EndJit_FHE, ApplyLookupTableWithPrecision,
                         ::testing::Values(1, 2, 3, 4, 5, 6, 7));

TEST(End2EndJit_FHE, apply_lookup_table_multiple_precision) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<6>, %arg1: !FHE.eint<3>) -> !FHE.eint<6> {
   %tlu_7 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
   %tlu_3 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
   %a = "FHE.apply_lookup_table"(%arg0, %tlu_7): (!FHE.eint<6>, tensor<64xi64>) -> (!FHE.eint<6>)
   %b = "FHE.apply_lookup_table"(%arg1, %tlu_3): (!FHE.eint<3>, tensor<8xi64>) -> (!FHE.eint<6>)
   %a_plus_b = "FHE.add_eint"(%a, %b): (!FHE.eint<6>, !FHE.eint<6>) -> (!FHE.eint<6>)
   return %a_plus_b: !FHE.eint<6>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(23_u64, 7_u64), 30);
}

TEST(End2EndJit_FHE, apply_lookup_table_random_func) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !FHE.eint<6>) -> !FHE.eint<6> {
   %tlu = arith.constant dense<[16, 91, 16, 83, 80, 74, 21, 96, 1, 63, 49, 122, 76, 89, 74, 55, 109, 110, 103, 54, 105, 14, 66, 47, 52, 89, 7, 10, 73, 44, 119, 92, 25, 104, 123, 100, 108, 86, 29, 121, 118, 52, 107, 48, 34, 37, 13, 122, 107, 48, 74, 59, 96, 36, 50, 55, 120, 72, 27, 45, 12, 5, 96, 12]> : tensor<64xi64>
   %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<6>, tensor<64xi64>) -> (!FHE.eint<6>)
   return %1: !FHE.eint<6>
}
)XXX");

  ASSERT_EXPECTED_VALUE(lambda(5_u64), 74);
  ASSERT_EXPECTED_VALUE(lambda(62_u64), 96);
  ASSERT_EXPECTED_VALUE(lambda(0_u64), 16);
  ASSERT_EXPECTED_VALUE(lambda(63_u64), 12);
}