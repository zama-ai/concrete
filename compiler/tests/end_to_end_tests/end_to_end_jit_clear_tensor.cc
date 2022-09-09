#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"

///////////////////////////////////////////////////////////////////////////////
// 1D tensor //////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_ClearTensor_2D, constant_i8) {
  checkedJit(lambda,
             R"XXX(
func.func @main() -> tensor<2x2xi8> {
  %cst = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi8>
  return %cst : tensor<2x2xi8>
}
)XXX",
             "main", true);

  llvm::Expected<std::vector<uint8_t>> res =
      lambda.operator()<std::vector<uint8_t>>();

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)4);

  EXPECT_EQ((*res)[0], 0);
  EXPECT_EQ((*res)[1], 1);
  EXPECT_EQ((*res)[2], 2);
  EXPECT_EQ((*res)[3], 3);
}

TEST(End2EndJit_ClearTensor_1D, identity) {
  checkedJit(lambda,
             R"XXX(
func.func @main(%t: tensor<10xi64>) -> tensor<10xi64> {
  return %t : tensor<10xi64>
}
)XXX",
             "main", true);

  uint64_t arg[]{0xFFFFFFFFFFFFFFFF,
                 0,
                 8978,
                 2587490,
                 90,
                 197864,
                 698735,
                 72132,
                 87474,
                 42};

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>(arg, ARRAY_SIZE(arg));

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)10);

  for (size_t i = 0; i < res->size(); i++) {
    EXPECT_EQ(arg[i], res->operator[](i)) << "result differ at index " << i;
  }
}

TEST(End2EndJit_ClearTensor_1D, identity_i8) {
  checkedJit(lambda,
             R"XXX(
func.func @main(%t: tensor<10xi8>) -> tensor<10xi8> {
  return %t : tensor<10xi8>
}
)XXX",
             "main", true);

  uint8_t arg[]{16, 21, 3, 127, 9, 17, 32, 18, 29, 104};

  llvm::Expected<std::vector<uint8_t>> res =
      lambda.operator()<std::vector<uint8_t>>(arg, ARRAY_SIZE(arg));

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)10);

  for (size_t i = 0; i < res->size(); i++) {
    EXPECT_EQ(arg[i], res->operator[](i)) << "result differ at index " << i;
  }
}

TEST(End2EndJit_ClearTensor_1D, extract_64) {
  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<10xi64>, %i: index) -> i64{
  %c = tensor.extract %t[%i] : tensor<10xi64>
  return %c : i64
}
)XXX",
             "main", true);

  uint64_t arg[]{0xFFFFFFFFFFFFFFFF,
                 0,
                 8978,
                 2587490,
                 90,
                 197864,
                 698735,
                 72132,
                 87474,
                 42};

  for (size_t i = 0; i < ARRAY_SIZE(arg); i++) {
    ASSERT_EXPECTED_VALUE(lambda(arg, ARRAY_SIZE(arg), i), arg[i]);
  }
}

TEST(End2EndJit_ClearTensor_1D, extract_32) {
  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<10xi32>, %i: index) -> i32{
  %c = tensor.extract %t[%i] : tensor<10xi32>
  return %c : i32
}
)XXX",
             "main", true);

  uint32_t arg[]{0xFFFFFFFF, 0,      8978,  2587490, 90,
                 197864,     698735, 72132, 87474,   42};

  for (size_t i = 0; i < ARRAY_SIZE(arg); i++) {
    ASSERT_EXPECTED_VALUE(lambda(arg, ARRAY_SIZE(arg), i), arg[i]);
  }
}

TEST(End2EndJit_ClearTensor_1D, extract_16) {

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<10xi16>, %i: index) -> i16{
  %c = tensor.extract %t[%i] : tensor<10xi16>
  return %c : i16
}
)XXX",
             "main", true);

  uint16_t arg[]{0xFFFF, 0,     59589, 47826, 16227,
                 63269,  36435, 52380, 7401,  13313};

  for (size_t i = 0; i < ARRAY_SIZE(arg); i++) {
    ASSERT_EXPECTED_VALUE(lambda(arg, ARRAY_SIZE(arg), i), arg[i]);
  }
}

TEST(End2EndJit_ClearTensor_1D, extract_8) {

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<10xi8>, %i: index) -> i8{
  %c = tensor.extract %t[%i] : tensor<10xi8>
  return %c : i8
}
)XXX",
             "main", true);

  uint8_t arg[]{0xFF, 0, 120, 225, 14, 177, 131, 84, 174, 93};

  for (size_t i = 0; i < ARRAY_SIZE(arg); i++) {
    ASSERT_EXPECTED_VALUE(lambda(arg, ARRAY_SIZE(arg), i), arg[i]);
  }
}

TEST(End2EndJit_ClearTensor_1D, extract_5) {

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<10xi5>, %i: index) -> i5{
  %c = tensor.extract %t[%i] : tensor<10xi5>
  return %c : i5
}
)XXX",
             "main", true);

  uint8_t arg[]{32, 0, 10, 25, 14, 25, 18, 28, 14, 7};

  for (size_t i = 0; i < ARRAY_SIZE(arg); i++) {
    ASSERT_EXPECTED_VALUE(lambda(arg, ARRAY_SIZE(arg), i), arg[i]);
  }
}

TEST(End2EndJit_ClearTensor_1D, extract_1) {

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<10xi1>, %i: index) -> i1{
  %c = tensor.extract %t[%i] : tensor<10xi1>
  return %c : i1
}
)XXX",
             "main", true);

  uint8_t arg[]{0, 0, 1, 0, 1, 1, 0, 1, 1, 0};

  for (size_t i = 0; i < ARRAY_SIZE(arg); i++) {
    ASSERT_EXPECTED_VALUE(lambda(arg, ARRAY_SIZE(arg), i), arg[i]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// 2D tensor
//////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

const size_t numDim = 2;
const size_t dim0 = 2;
const size_t dim1 = 10;
const int64_t dims[numDim]{dim0, dim1};
static std::vector<uint64_t> tensor2D{
    0xFFFFFFFFFFFFFFFF,
    0,
    8978,
    2587490,
    90,
    197864,
    698735,
    72132,
    87474,
    42,
    986,
    1873,
    298493,
    34939,
    443,
    59874,
    43,
    743,
    8409,
    9433,
};
const llvm::ArrayRef<int64_t> shape2D(dims, numDim);
#define GET_2D(tensor, i, j) (tensor)[i * dims[1] + j]

#define TENSOR2D_GET(i, j) GET_2D(tensor2D, i, j)

TEST(End2EndJit_ClearTensor_2D, DISABLED_identity) {

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<2x10xi64>) -> tensor<2x10xi64> {
  return %t : tensor<2x10xi64>
}
)XXX",
             "main", true);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>
      arg(tensor2D, shape2D);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), tensor2D.size());

  for (size_t i = 0; i < tensor2D.size(); i++) {
    EXPECT_EQ(tensor2D[i], (*res)[i]) << "result differ at pos " << i;
  }
}

TEST(End2EndJit_ClearTensor_2D, extract) {

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<2x10xi64>, %i: index, %j: index) -> i64 {
  %c = tensor.extract %t[%i, %j] : tensor<2x10xi64>
  return %c : i64
}
)XXX",
             "main", true);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>
      arg(tensor2D, shape2D);

  for (int64_t i = 0; i < dims[0]; i++) {
    for (int64_t j = 0; j < dims[1]; j++) {
      mlir::concretelang::IntLambdaArgument<size_t> argi(i);
      mlir::concretelang::IntLambdaArgument<size_t> argj(j);
      ASSERT_EXPECTED_VALUE(lambda({&arg, &argi, &argj}), TENSOR2D_GET(i, j));
    }
  }
}

TEST(End2EndJit_ClearTensor_2D, extract_slice) {

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<2x10xi64>) -> tensor<1x5xi64> {
  %r = tensor.extract_slice %t[1, 5][1, 5][1, 1] : tensor<2x10xi64> to
  tensor<1x5xi64> return %r : tensor<1x5xi64>
}
)XXX",
             "main", true);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>
      arg(tensor2D, shape2D);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)1 * 5);

  // Check the sub slice
  for (size_t j = 0; j < 5; j++) {
    // Get and assert the result
    ASSERT_EQ((*res)[j], TENSOR2D_GET(1, j + 5));
  }
}

TEST(End2EndJit_ClearTensor_2D, extract_slice_stride) {

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<2x10xi64>) -> tensor<1x5xi64> {
  %r = tensor.extract_slice %t[1, 0][1, 5][1, 2] : tensor<2x10xi64> to
  tensor<1x5xi64> return %r : tensor<1x5xi64>
}
)XXX",
             "main", true);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>
      arg(tensor2D, shape2D);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)1 * 5);

  // Check the sub slice
  for (size_t j = 0; j < 5; j++) {
    // Get and assert the result
    ASSERT_EQ((*res)[j], TENSOR2D_GET(1, j * 2));
  }
}

TEST(End2EndJit_ClearTensor_2D, insert_slice) {

  checkedJit(lambda, R"XXX(
func.func @main(%t0: tensor<2x10xi64>, %t1: tensor<2x2xi64>) -> tensor<2x10xi64> {
  %r = tensor.insert_slice %t1 into %t0[0, 5][2, 2][1, 1] : tensor<2x2xi64>
  into tensor<2x10xi64> return %r : tensor<2x10xi64>
}
)XXX",
             "main", true);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>
      t0(tensor2D, shape2D);
  int64_t t1Shape[] = {2, 2};
  uint64_t t1Buffer[]{6, 9, 4, 0};
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>
      t1(t1Buffer, t1Shape);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&t0, &t1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), tensor2D.size());

  // Check the sub slice
  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      if (j < 5 || j >= 5 + 2) {
        ASSERT_EQ(GET_2D(*res, i, j), TENSOR2D_GET(i, j))
            << "at indexes (" << i << "," << j << ")";
      } else {
        // Get and assert the result
        ASSERT_EQ(GET_2D(*res, i, j), t1Buffer[i * 2 + j - 5])
            << "at indexes (" << i << "," << j << ")";
        ;
      }
    }
  }
}

template <typename T>
void checkResultTensor(
    bool &status,
    llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> &res) {
  status = false;

  ASSERT_TRUE((*res)
                  ->isa<mlir::concretelang::TensorLambdaArgument<
                      mlir::concretelang::IntLambdaArgument<T>>>());

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<T>> &resp =
      (*res)
          ->cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<T>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)3);
  ASSERT_EQ(resp.getDimensions().at(0), 5);
  ASSERT_EQ(resp.getDimensions().at(1), 3);
  ASSERT_EQ(resp.getDimensions().at(2), 2);

  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 5 * 3 * 2);

  for (size_t i = 0; i < 5 * 3 * 2; i++) {
    ASSERT_EQ(resp.getValue()[i], 1_u64);
  }

  status = true;
}

class ReturnTensorWithPrecision : public ::testing::TestWithParam<int> {};

TEST_P(ReturnTensorWithPrecision, return_tensor) {
  uint64_t precision = GetParam();
  std::ostringstream mlirProgram;

  mlirProgram << "func.func @main() -> tensor<5x3x2xi" << precision << "> {\n"
              << "  %res = arith.constant dense<1> : tensor<5x3x2xi"
              << precision << ">\n"
              << "   return %res : tensor<5x3x2xi" << precision << ">\n"
              << "}";

  checkedJit(lambda, mlirProgram.str(), "main", true);

  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
          {});
  ASSERT_EXPECTED_SUCCESS(res);
  bool status;

  if (precision > 64)
    GTEST_FATAL_FAILURE_("Cannot handle precision > 64 bits");
  else if (precision > 32)
    checkResultTensor<uint64_t>(status, res);
  else if (precision > 16)
    checkResultTensor<uint32_t>(status, res);
  else if (precision > 8)
    checkResultTensor<uint16_t>(status, res);
  else
    checkResultTensor<uint8_t>(status, res);

  ASSERT_TRUE(status);
}

INSTANTIATE_TEST_SUITE_P(DISABLED_ReturnTensor, ReturnTensorWithPrecision,
                         ::testing::Values(1, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                                           63, 64));
