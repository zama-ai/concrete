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

  for (int64_t i = 0; i < dims[0]; i++) {
    for (int64_t j = 0; j < dims[1]; j++) {
      mlir::zamalang::IntLambdaArgument<size_t> argi(i);
      mlir::zamalang::IntLambdaArgument<size_t> argj(j);
      ASSERT_EXPECTED_VALUE(lambda({&arg, &argi, &argj}), TENSOR2D_GET(i, j));
    }
  }
}

TEST(End2EndJit_EncryptedTensor_2D, extract_slice) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<2x10x!HLFHE.eint<6>>) -> tensor<1x5x!HLFHE.eint<6>> {
  %r = tensor.extract_slice %t[1, 5][1, 5][1, 1] : tensor<2x10x!HLFHE.eint<6>> to tensor<1x5x!HLFHE.eint<6>>
  return %r : tensor<1x5x!HLFHE.eint<6>>
}
)XXX");

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
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

TEST(End2EndJit_EncryptedTensor_2D, extract_slice_parametric_2x2) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<8x4x!HLFHE.eint<6>>, %y: index, %x: index) -> tensor<2x2x!HLFHE.eint<6>> {
  %r = tensor.extract_slice %t[%y, %x][2, 2][1, 1] : tensor<8x4x!HLFHE.eint<6>> to tensor<2x2x!HLFHE.eint<6>>
  return %r : tensor<2x2x!HLFHE.eint<6>>
}
)XXX");
  const size_t rows = 8;
  const size_t cols = 4;
  const size_t tileSize = 2;
  const uint8_t A[rows][cols] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 0, 1, 2},
                                 {3, 4, 5, 6}, {7, 8, 9, 0}, {1, 2, 3, 4},
                                 {5, 6, 7, 8}, {9, 0, 1, 2}};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      argT(llvm::ArrayRef<uint8_t>((const uint8_t *)A, rows * cols),
           {rows, cols});

  for (uint64_t y = 0; y <= rows - tileSize; y += tileSize) {
    for (uint64_t x = 0; x <= cols - tileSize; x += tileSize) {
      mlir::zamalang::IntLambdaArgument<uint64_t> argY(y);
      mlir::zamalang::IntLambdaArgument<uint64_t> argX(x);

      llvm::Expected<std::vector<uint64_t>> res =
          lambda.operator()<std::vector<uint64_t>>({&argT, &argY, &argX});

      ASSERT_EXPECTED_SUCCESS(res);
      ASSERT_EQ(res->size(), tileSize * tileSize);
      ASSERT_EQ((*res)[0], A[y][x]);
      ASSERT_EQ((*res)[1], A[y][x + 1]);
      ASSERT_EQ((*res)[2], A[y + 1][x]);
      ASSERT_EQ((*res)[3], A[y + 1][x + 1]);
    }
  }
}

// Extracts 4D tiles from a 4D tensor
TEST(End2EndJit_EncryptedTensor_4D, extract_slice_parametric_2x2x2x2) {
  constexpr int64_t dimSizes[4] = {8, 4, 5, 3};

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%t: tensor<8x4x5x3x!HLFHE.eint<6>>, %d0: index, %d1: index, %d2: index, %d3: index) -> tensor<2x2x2x2x!HLFHE.eint<6>> {
  %r = tensor.extract_slice %t[%d0, %d1, %d2, %d3][2, 2, 2, 2][1, 1, 1, 1] : tensor<8x4x5x3x!HLFHE.eint<6>> to tensor<2x2x2x2x!HLFHE.eint<6>>
  return %r : tensor<2x2x2x2x!HLFHE.eint<6>>
}
)XXX");
  uint8_t A[dimSizes[0]][dimSizes[1]][dimSizes[2]][dimSizes[3]];

  // Fill with some reproducible pattern
  for (int64_t d0 = 0; d0 < dimSizes[0]; d0++) {
    for (int64_t d1 = 0; d1 < dimSizes[1]; d1++) {
      for (int64_t d2 = 0; d2 < dimSizes[2]; d2++) {
        for (int64_t d3 = 0; d3 < dimSizes[3]; d3++) {
          A[d0][d1][d2][d3] = d0 + d1 + d2 + d3;
        }
      }
    }
  }

  const size_t ncoords = 5;
  const size_t coords[ncoords][4] = {
      {0, 0, 0, 0}, {1, 1, 1, 1}, {6, 2, 0, 1}, {3, 1, 2, 0}, {3, 1, 2, 1}};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      argT(llvm::ArrayRef<uint8_t>((const uint8_t *)A,
                                   dimSizes[0] * dimSizes[1] * dimSizes[2] *
                                       dimSizes[3]),
           dimSizes);

  for (uint64_t i = 0; i < ncoords; i++) {
    size_t d0 = coords[i][0];
    size_t d1 = coords[i][1];
    size_t d2 = coords[i][2];
    size_t d3 = coords[i][3];

    mlir::zamalang::IntLambdaArgument<uint64_t> argD0(d0);
    mlir::zamalang::IntLambdaArgument<uint64_t> argD1(d1);
    mlir::zamalang::IntLambdaArgument<uint64_t> argD2(d2);
    mlir::zamalang::IntLambdaArgument<uint64_t> argD3(d3);

    llvm::Expected<std::vector<uint64_t>> res =
        lambda.operator()<std::vector<uint64_t>>(
            {&argT, &argD0, &argD1, &argD2, &argD3});

    ASSERT_EXPECTED_SUCCESS(res);
    ASSERT_EQ(res->size(), (size_t)(2 * 2 * 2 * 2));

    for (size_t rd0 = 0; rd0 < 2; rd0++) {
      for (size_t rd1 = 0; rd1 < 2; rd1++) {
        for (size_t rd2 = 0; rd2 < 2; rd2++) {
          for (size_t rd3 = 0; rd3 < 2; rd3++) {
            ASSERT_EQ((*res)[rd0 * 8 + rd1 * 4 + rd2 * 2 + rd3],
                      A[d0 + rd0][d1 + rd1][d2 + rd2][d3 + rd3]);
          }
        }
      }
    }
  }
}

TEST(End2EndJit_EncryptedTensor_2D, extract_slice_stride) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%t: tensor<2x10x!HLFHE.eint<6>>) -> tensor<1x5x!HLFHE.eint<6>> {
  %r = tensor.extract_slice %t[1, 0][1, 5][1, 2] : tensor<2x10x!HLFHE.eint<6>> to tensor<1x5x!HLFHE.eint<6>>
  return %r : tensor<1x5x!HLFHE.eint<6>>
}
)XXX");

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
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

TEST(End2EndJit_EncryptedTensor_2D, insert_slice) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%t0: tensor<2x10x!HLFHE.eint<6>>, %t1: tensor<2x2x!HLFHE.eint<6>>)
-> tensor<2x10x!HLFHE.eint<6>> {
  %r = tensor.insert_slice %t1 into %t0[0, 5][2, 2][1, 1] : tensor<2x2x!HLFHE.eint<6>> into tensor<2x10x!HLFHE.eint<6>>
  return %r : tensor<2x10x!HLFHE.eint<6>>
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
