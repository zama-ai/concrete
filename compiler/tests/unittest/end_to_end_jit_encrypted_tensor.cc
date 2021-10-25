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

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg add_eint ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, add_eint_term_to_term) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x!HLFHE.eint<4>>, %a1: tensor<4x!HLFHE.eint<4>>) -> tensor<4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<4x!HLFHE.eint<4>>, tensor<4x!HLFHE.eint<4>>) -> tensor<4x!HLFHE.eint<4>>
    return %res : tensor<4x!HLFHE.eint<4>>
  }
)XXX";
  const uint8_t a0[4]{31, 6, 12, 9};
  const uint8_t a1[4]{32, 9, 2, 3};

  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));

  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %a0 and %a1 argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)a0, 4));
  ASSERT_LLVM_ERROR(argument->setArg(1, (uint8_t *)a1, 4));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[4];
  ASSERT_LLVM_ERROR(argument->getResult(0, (uint64_t *)result, 4));
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(result[i], a0[i] + a1[i])
        << "result differ at pos " << i << ", expect " << a0[i] + a1[i]
        << " got " << result[i];
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_term_to_term_broadcast) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x1x4x!HLFHE.eint<4>>, %a1: tensor<1x4x4x!HLFHE.eint<4>>) -> tensor<4x4x4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<4x1x4x!HLFHE.eint<4>>, tensor<1x4x4x!HLFHE.eint<4>>) -> tensor<4x4x4x!HLFHE.eint<4>>
    return %res : tensor<4x4x4x!HLFHE.eint<4>>
  }
)XXX";
  const uint8_t a0[4][1][4]{
      {{1, 2, 3, 4}},
      {{5, 6, 7, 8}},
      {{9, 10, 11, 12}},
      {{13, 14, 15, 16}},
  };
  const uint8_t a1[1][4][4]{
      {
          {1, 2, 3, 4},
          {5, 6, 7, 8},
          {9, 10, 11, 12},
          {13, 14, 15, 16},
      },
  };

  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));

  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %a0 and %a1 argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)a0, {4, 1, 4}));
  ASSERT_LLVM_ERROR(argument->setArg(1, (uint8_t *)a1, {1, 4, 4}));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[4][4][4];
  ASSERT_LLVM_ERROR(argument->getResult(0, (uint64_t *)result, 4 * 4 * 4));
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ(result[i][j][k], a0[i][0][k] + a1[0][j][k])
            << "result differ at pos " << i << ", expect "
            << a0[i][0][k] + a1[0][j][k] << " got " << result[i];
      }
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_matrix_column) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
  //
  // [1,2,3]   [1]   [2,3,4]
  // [4,5,6] + [2] = [6,7,8]
  // [7,8,9]   [3]   [10,11,12]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
    return %res : tensor<3x3x!HLFHE.eint<4>>
  }
)XXX";
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[3][1]{
      {1},
      {2},
      {3},
  };

  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));

  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %a0 and %a1 argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)a0, {3, 3}));
  ASSERT_LLVM_ERROR(argument->setArg(1, (uint8_t *)a1, {3, 1}));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[3][3];
  ASSERT_LLVM_ERROR(argument->getResult(0, (uint64_t *)result, 3 * 3));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(result[i][j], a0[i][j] + a1[i][0])
          << "result differ at pos " << i << ", expect " << a0[i][j] + a1[i][0]
          << " got " << result[i];
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_matrix_line) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
  //
  // [1,2,3]             [2,4,6]
  // [4,5,6] + [1,2,3] = [5,7,9]
  // [7,8,9]             [8,10,12]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
    return %res : tensor<3x3x!HLFHE.eint<4>>
  }
)XXX";
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));

  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %a0 and %a1 argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)a0, {3, 3}));
  ASSERT_LLVM_ERROR(argument->setArg(1, (uint8_t *)a1, {1, 3}));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[3][3];
  ASSERT_LLVM_ERROR(argument->getResult(0, (uint64_t *)result, 3 * 3));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(result[i][j], a0[i][j] + a1[0][j])
          << "result differ at pos (" << i << "," << j << "), expect "
          << a0[i][j] + a1[0][j] << " got " << result[i][j] << "\n";
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_matrix_line_missing_dim) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
    return %res : tensor<3x3x!HLFHE.eint<4>>
  }
)XXX";
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));

  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %a0 and %a1 argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint8_t *)a0, {3, 3}));
  ASSERT_LLVM_ERROR(argument->setArg(1, (uint8_t *)a1, {3}));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[3][3];
  ASSERT_LLVM_ERROR(argument->getResult(0, (uint64_t *)result, 3 * 3));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(result[i][j], a0[i][j] + a1[0][j])
          << "result differ at pos (" << i << "," << j << "), expect "
          << a0[i][j] + a1[0][j] << " got " << result[i][j] << "\n";
    }
  }
}