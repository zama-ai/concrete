#include "end_to_end_jit_test.h"

///////////////////////////////////////////////////////////////////////////////
// 2D encrypted tensor ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

const size_t numDim = 2;
const int64_t dim0 = 2;
const int64_t dim1 = 10;
const int64_t dims[numDim]{dim0, dim1};
const uint8_t tensor2D[dim0][dim1]{
    {63, 12, 7, 43, 52, 9, 26, 34, 22, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
};
const llvm::ArrayRef<int64_t> shape2D(dims, numDim);

TEST(End2EndJit_EncryptedTensor_2D, identity) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<2x10x!HLFHE.eint<6>>) -> tensor<2x10x!HLFHE.eint<6>> {
  return %t : tensor<2x10x!HLFHE.eint<6>>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));

  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)tensor2D, shape2D));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[dims[0]][dims[1]];
  ASSERT_LLVM_ERROR(
      argument->getResult(0, (uint64_t *)result, dims[0] * dims[1]));
  for (size_t i = 0; i < dims[0]; i++) {
    for (size_t j = 0; j < dims[1]; j++) {
      EXPECT_EQ(tensor2D[i][j], result[i][j])
          << "result differ at pos " << i << "," << j;
    }
  }
}

TEST(End2EndJit_EncryptedTensor_2D, extract) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<2x10x!HLFHE.eint<6>>, %i: index, %j: index) -> !HLFHE.eint<6> {
  %c = tensor.extract %t[%i, %j] : tensor<2x10x!HLFHE.eint<6>>
  return %c : !HLFHE.eint<6>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)tensor2D, shape2D));
  for (size_t i = 0; i < dims[0]; i++) {
    for (size_t j = 0; j < dims[1]; j++) {
      // Set %i, %j
      ASSERT_LLVM_ERROR(argument->setArg(1, i));
      ASSERT_LLVM_ERROR(argument->setArg(2, j));
      // Invoke the function
      ASSERT_LLVM_ERROR(engine.invoke(*argument));
      // Get and assert the result
      uint64_t res = 0;
      ASSERT_LLVM_ERROR(argument->getResult(0, res));
      ASSERT_EQ(res, tensor2D[i][j]);
    }
  }
}

TEST(End2EndJit_EncryptedTensor_2D, extract_slice) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<2x10x!HLFHE.eint<6>>) -> tensor<1x5x!HLFHE.eint<6>> {
  %r = tensor.extract_slice %t[1, 5][1, 5][1, 1] : tensor<2x10x!HLFHE.eint<6>> to tensor<1x5x!HLFHE.eint<6>>
  return %r : tensor<1x5x!HLFHE.eint<6>>
}
)XXX";

  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)tensor2D, shape2D));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[1][5];
  ASSERT_LLVM_ERROR(argument->getResult(0, (uint64_t *)result, 1 * 5));
  // Check the sub slice
  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 5; j++) {
      // Get and assert the result
      ASSERT_EQ(result[i][j], tensor2D[i + 1][j + 5]);
    }
  }
}

TEST(End2EndJit_EncryptedTensor_2D, extract_slice_stride) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<2x10x!HLFHE.eint<6>>) -> tensor<1x5x!HLFHE.eint<6>> {
  %r = tensor.extract_slice %t[1, 0][1, 5][1, 2] : tensor<2x10x!HLFHE.eint<6>> to tensor<1x5x!HLFHE.eint<6>>
  return %r : tensor<1x5x!HLFHE.eint<6>>
}
)XXX";

  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)tensor2D, shape2D));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[1][5];
  ASSERT_LLVM_ERROR(argument->getResult(0, (uint64_t *)result, 1 * 5));
  // Check the sub slice
  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 5; j++) {
      // Get and assert the result
      ASSERT_EQ(result[i][j], tensor2D[i + 1][j * 2]);
    }
  }
}

TEST(End2EndJit_EncryptedTensor_2D, insert_slice) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t0: tensor<2x10x!HLFHE.eint<6>>, %t1: tensor<2x2x!HLFHE.eint<6>>) -> tensor<2x10x!HLFHE.eint<6>> {
  %r = tensor.insert_slice %t1 into %t0[0, 5][2, 2][1, 1] : tensor<2x2x!HLFHE.eint<6>> into tensor<2x10x!HLFHE.eint<6>>
  return %r : tensor<2x10x!HLFHE.eint<6>>
}
)XXX";

  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t0 argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)tensor2D, shape2D));
  // Set the %t1 argument
  int64_t t1_dim[2] = {2, 2};
  uint8_t t1[2][2]{{6, 9}, {4, 0}};
  ASSERT_LLVM_ERROR(
      argument->setArg(1, (uint8_t *)t1, llvm::ArrayRef<int64_t>(t1_dim, 2)));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[dim0][dim1];
  ASSERT_LLVM_ERROR(argument->getResult(0, (uint64_t *)result, dim0 * dim1));
  // Check the sub slice
  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      if (j < 5 || j >= 5 + 2) {
        ASSERT_EQ(result[i][j], tensor2D[i][j])
            << "at indexes (" << i << "," << j << ")";
      } else {
        // Get and assert the result
        ASSERT_EQ(result[i][j], t1[i][j - 5])
            << "at indexes (" << i << "," << j << ")";
        ;
      }
    }
  }
}
