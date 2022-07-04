#include "end_to_end_jit_test.h"

TEST(End2EndJit_EncryptedTensor_2D, extract_slice_parametric_2x2) {
  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<8x4x!FHE.eint<6>>, %y: index, %x: index) -> tensor<2x2x!FHE.eint<6>> {
  %r = tensor.extract_slice %t[%y, %x][2, 2][1, 1] : tensor<8x4x!FHE.eint<6>> to tensor<2x2x!FHE.eint<6>>
  return %r : tensor<2x2x!FHE.eint<6>>
}
)XXX");
  const size_t rows = 8;
  const size_t cols = 4;
  const size_t tileSize = 2;
  const uint8_t A[rows][cols] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 0, 1, 2},
                                 {3, 4, 5, 6}, {7, 8, 9, 0}, {1, 2, 3, 4},
                                 {5, 6, 7, 8}, {9, 0, 1, 2}};

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      argT(llvm::ArrayRef<uint8_t>((const uint8_t *)A, rows * cols),
           {rows, cols});

  for (uint64_t y = 0; y <= rows - tileSize; y += tileSize) {
    for (uint64_t x = 0; x <= cols - tileSize; x += tileSize) {
      mlir::concretelang::IntLambdaArgument<uint64_t> argY(y);
      mlir::concretelang::IntLambdaArgument<uint64_t> argX(x);

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

  checkedJit(lambda, R"XXX(
func.func @main(%t: tensor<8x4x5x3x!FHE.eint<6>>, %d0: index, %d1: index, %d2: index, %d3: index) -> tensor<2x2x2x2x!FHE.eint<6>> {
  %r = tensor.extract_slice %t[%d0, %d1, %d2, %d3][2, 2, 2, 2][1, 1, 1, 1] : tensor<8x4x5x3x!FHE.eint<6>> to tensor<2x2x2x2x!FHE.eint<6>>
  return %r : tensor<2x2x2x2x!FHE.eint<6>>
}
)XXX");
  uint8_t A[dimSizes[0]][dimSizes[1]][dimSizes[2]][dimSizes[3]];

  // Fill with some reproducible pattern
  for (size_t d0 = 0; d0 < dimSizes[0]; d0++) {
    for (size_t d1 = 0; d1 < dimSizes[1]; d1++) {
      for (size_t d2 = 0; d2 < dimSizes[2]; d2++) {
        for (size_t d3 = 0; d3 < dimSizes[3]; d3++) {
          A[d0][d1][d2][d3] = d0 + d1 + d2 + d3;
        }
      }
    }
  }

  const size_t ncoords = 5;
  const size_t coords[ncoords][4] = {
      {0, 0, 0, 0}, {1, 1, 1, 1}, {6, 2, 0, 1}, {3, 1, 2, 0}, {3, 1, 2, 1}};

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      argT(llvm::ArrayRef<uint8_t>((const uint8_t *)A,
                                   dimSizes[0] * dimSizes[1] * dimSizes[2] *
                                       dimSizes[3]),
           dimSizes);

  for (uint64_t i = 0; i < ncoords; i++) {
    size_t d0 = coords[i][0];
    size_t d1 = coords[i][1];
    size_t d2 = coords[i][2];
    size_t d3 = coords[i][3];

    mlir::concretelang::IntLambdaArgument<uint64_t> argD0(d0);
    mlir::concretelang::IntLambdaArgument<uint64_t> argD1(d1);
    mlir::concretelang::IntLambdaArgument<uint64_t> argD2(d2);
    mlir::concretelang::IntLambdaArgument<uint64_t> argD3(d3);

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
