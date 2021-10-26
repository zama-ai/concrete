#include "end_to_end_jit_test.h"

///////////////////////////////////////////////////////////////////////////////
// 2D encrypted tensor ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

const size_t numDim = 2;
const int64_t dim0 = 2;
const int64_t dim1 = 10;
const int64_t dims[numDim]{dim0, dim1};
static std::vector<uint8_t> tensor2D{
    63, 12, 7, 43, 52, 9, 26, 34, 22, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
};
const llvm::ArrayRef<int64_t> shape2D(dims, numDim);
#define GET_2D(tensor, i, j) (tensor)[i * dims[1] + j]

#define TENSOR2D_GET(i, j) GET_2D(tensor2D, i, j)

TEST(End2EndJit_EncryptedTensor_2D, identity) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%t: tensor<2x10x!HLFHE.eint<6>>) -> tensor<2x10x!HLFHE.eint<6>> {
  return %t : tensor<2x10x!HLFHE.eint<6>>
}
)XXX");

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg(tensor2D, shape2D);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), tensor2D.size());

  for (size_t i = 0; i < tensor2D.size(); i++) {
    EXPECT_EQ(tensor2D[i], (*res)[i]) << "result differ at pos " << i;
  }
}

TEST(End2EndJit_EncryptedTensor_2D, extract) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<2x10x!HLFHE.eint<6>>, %i: index, %j: index) ->
!HLFHE.eint<6> {
  %c = tensor.extract %t[%i, %j] : tensor<2x10x!HLFHE.eint<6>>
  return %c : !HLFHE.eint<6>
}
)XXX");

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg(tensor2D, shape2D);

  for (size_t i = 0; i < dims[0]; i++) {
    for (size_t j = 0; j < dims[1]; j++) {
      auto pos = i * dims[1] + j;
      mlir::zamalang::IntLambdaArgument<size_t> argi(i);
      mlir::zamalang::IntLambdaArgument<size_t> argj(j);
      ASSERT_EXPECTED_VALUE(lambda({&arg, &argi, &argj}), TENSOR2D_GET(i, j));
    }
  }
}

TEST(End2EndJit_EncryptedTensor_2D, extract_slice) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<2x10x!HLFHE.eint<6>>) -> tensor<1x5x!HLFHE.eint<6>> {
  %r = tensor.extract_slice %t[1, 5][1, 5][1, 1] :
  tensor<2x10x!HLFHE.eint<6>> to tensor<1x5x!HLFHE.eint<6>> return %r :
  tensor<1x5x!HLFHE.eint<6>>
}
)XXX");

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg(tensor2D, shape2D);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 1 * 5);

  // Check the sub slice
  for (size_t j = 0; j < 5; j++) {
    // Get and assert the result
    ASSERT_EQ((*res)[j], TENSOR2D_GET(1, j + 5));
  }
}

TEST(End2EndJit_EncryptedTensor_2D, extract_slice_stride) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%t: tensor<2x10x!HLFHE.eint<6>>) -> tensor<1x5x!HLFHE.eint<6>> {
  %r = tensor.extract_slice %t[1, 0][1, 5][1, 2] :
  tensor<2x10x!HLFHE.eint<6>> to tensor<1x5x!HLFHE.eint<6>> return %r :
  tensor<1x5x!HLFHE.eint<6>>
}
)XXX");

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg(tensor2D, shape2D);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 1 * 5);

  // Check the sub slice
  for (size_t j = 0; j < 5; j++) {
    // Get and assert the result
    ASSERT_EQ((*res)[j], TENSOR2D_GET(1, j * 2));
  }
}

TEST(End2EndJit_EncryptedTensor_2D, insert_slice) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%t0: tensor<2x10x!HLFHE.eint<6>>, %t1: tensor<2x2x!HLFHE.eint<6>>)
-> tensor<2x10x!HLFHE.eint<6>> {
  %r = tensor.insert_slice %t1 into %t0[0, 5][2, 2][1, 1] :
  tensor<2x2x!HLFHE.eint<6>> into tensor<2x10x!HLFHE.eint<6>> return %r :
  tensor<2x10x!HLFHE.eint<6>>
}
)XXX");

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      t0(tensor2D, shape2D);
  int64_t t1Shape[] = {2, 2};
  uint8_t t1Buffer[]{6, 9, 4, 0};
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
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