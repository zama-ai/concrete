#include "end_to_end_jit_test.h"

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg add_eint_int ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, add_eint_int_term_to_term) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x!HLFHE.eint<4>>, %a1: tensor<4xi5>) -> tensor<4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<4>>, tensor<4xi5>) -> tensor<4x!HLFHE.eint<4>>
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

TEST(End2EndJit_HLFHELinalg, add_eint_int_term_to_term_broadcast) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x1x4x!HLFHE.eint<4>>, %a1: tensor<1x4x4xi5>) -> tensor<4x4x4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x1x4x!HLFHE.eint<4>>, tensor<1x4x4xi5>) -> tensor<4x4x4x!HLFHE.eint<4>>
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

TEST(End2EndJit_HLFHELinalg, add_eint_int_matrix_column) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
  //
  // [1,2,3]   [1]   [2,3,4]
  // [4,5,6] + [2] = [6,7,8]
  // [7,8,9]   [3]   [10,11,12]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3x1xi5>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!HLFHE.eint<4>>
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

TEST(End2EndJit_HLFHELinalg, add_eint_int_matrix_line) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
  //
  // [1,2,3]             [2,4,6]
  // [4,5,6] + [1,2,3] = [5,7,9]
  // [7,8,9]             [8,10,12]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<1x3xi5>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!HLFHE.eint<4>>
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

TEST(End2EndJit_HLFHELinalg, add_eint_int_matrix_line_missing_dim) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>>
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

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg sub_int_eint ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, sub_int_eint_term_to_term) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the term to term substraction of `%a0` with `%a1`
  func @main(%a0: tensor<4xi5>, %a1: tensor<4x!HLFHE.eint<4>>) -> tensor<4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4xi5>, tensor<4x!HLFHE.eint<4>>) -> tensor<4x!HLFHE.eint<4>>
    return %res : tensor<4x!HLFHE.eint<4>>
  }
)XXX";
  const uint8_t a0[4]{32, 9, 12, 9};
  const uint8_t a1[4]{31, 6, 2, 3};

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
    EXPECT_EQ(result[i], a0[i] - a1[i])
        << "result differ at pos " << i << ", expect " << a0[i] - a1[i]
        << " got " << result[i];
  }
}

TEST(End2EndJit_HLFHELinalg, sub_int_eint_term_to_term_broadcast) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the term to term substraction of `%a0` with `%a1`, where dimensions equals to one are stretched.
  func @main(%a0: tensor<4x1x4xi5>, %a1: tensor<1x4x4x!HLFHE.eint<4>>) -> tensor<4x4x4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4x1x4xi5>, tensor<1x4x4x!HLFHE.eint<4>>) -> tensor<4x4x4x!HLFHE.eint<4>>
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
        uint8_t expected = a0[i][0][k] - a1[0][j][k];
        EXPECT_EQ(result[i][j][k], expected)
            << "result differ at pos (" << i << "," << j << "), expect "
            << expected << " got " << result[i];
      }
    }
  }
}

TEST(End2EndJit_HLFHELinalg, sub_int_eint_matrix_column) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the substraction of a 3x3 matrix of integers and a 3x1 matrix (a column) of encrypted integers.
  //
  // [1,2,3]   [1]   [0,2,3]
  // [4,5,6] - [2] = [2,3,4]
  // [7,8,9]   [3]   [4,5,6]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
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
      EXPECT_EQ(result[i][j], a0[i][j] - a1[i][0])
          << "result differ at pos (" << i << "," << j << "), expect "
          << a0[i][j] - a1[i][0] << " got " << result[i][j];
    }
  }
}

TEST(End2EndJit_HLFHELinalg, sub_int_eint_matrix_line) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the substraction of a 3x3 matrix of integers and a 1x3 matrix (a line) of encrypted integers.
  //
  // [1,2,3]             [0,0,0]
  // [4,5,6] + [1,2,3] = [3,3,3]
  // [7,8,9]             [6,6,6]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
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
      EXPECT_EQ(result[i][j], a0[i][j] - a1[0][j])
          << "result differ at pos (" << i << "," << j << "), expect "
          << a0[i][j] - a1[0][j] << " got " << result[i][j] << "\n";
    }
  }
}

TEST(End2EndJit_HLFHELinalg, sub_int_eint_matrix_line_missing_dim) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
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
      EXPECT_EQ(result[i][j], a0[i][j] - a1[0][j])
          << "result differ at pos (" << i << "," << j << "), expect "
          << a0[i][j] - a1[0][j] << " got " << result[i][j] << "\n";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg mul_eint_int ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, mul_eint_int_term_to_term) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the term to term multiplication of `%a0` with `%a1`
  func @main(%a0: tensor<4x!HLFHE.eint<4>>, %a1: tensor<4xi5>) -> tensor<4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<4>>, tensor<4xi5>) -> tensor<4x!HLFHE.eint<4>>
    return %res : tensor<4x!HLFHE.eint<4>>
  }
)XXX";
  const uint8_t a0[4]{31, 6, 12, 9};
  const uint8_t a1[4]{2, 3, 2, 3};

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
    EXPECT_EQ(result[i], a0[i] * a1[i])
        << "result differ at pos " << i << ", expect " << a0[i] * a1[i]
        << " got " << result[i];
  }
}

TEST(End2EndJit_HLFHELinalg, mul_eint_int_term_to_term_broadcast) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the term to term multiplication of `%a0` with `%a1`, where dimensions equals to one are stretched.
  func @main(%a0: tensor<4x1x4x!HLFHE.eint<4>>, %a1: tensor<1x4x4xi5>) -> tensor<4x4x4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x1x4x!HLFHE.eint<4>>, tensor<1x4x4xi5>) -> tensor<4x4x4x!HLFHE.eint<4>>
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
          {1, 2, 0, 1},
          {2, 0, 1, 2},
          {0, 1, 2, 0},
          {1, 2, 0, 1},
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
        EXPECT_EQ(result[i][j][k], a0[i][0][k] * a1[0][j][k])
            << "result differ at pos " << i << ", expect "
            << a0[i][0][k] * a1[0][j][k] << " got " << result[i];
      }
    }
  }
}

TEST(End2EndJit_HLFHELinalg, mul_eint_int_matrix_column) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the multiplication of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of integers.
  //
  // [1,2,3]   [1]   [1,2,3]
  // [4,5,6] * [2] = [8,10,18]
  // [7,8,9]   [3]   [21,24,27]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3x1xi5>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!HLFHE.eint<4>>
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
      EXPECT_EQ(result[i][j], a0[i][j] * a1[i][0])
          << "result differ at pos " << i << ", expect " << a0[i][j] * a1[i][0]
          << " got " << result[i];
    }
  }
}

TEST(End2EndJit_HLFHELinalg, mul_eint_int_matrix_line) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Returns the multiplication of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
  //
  // [1,2,3]             [2,4,6]
  // [4,5,6] * [1,2,3] = [5,7,9]
  // [7,8,9]             [8,10,12]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<1x3xi5>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!HLFHE.eint<4>>
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
      EXPECT_EQ(result[i][j], a0[i][j] * a1[0][j])
          << "result differ at pos (" << i << "," << j << "), expect "
          << a0[i][j] * a1[0][j] << " got " << result[i][j] << "\n";
    }
  }
}

TEST(End2EndJit_HLFHELinalg, mul_eint_int_matrix_line_missing_dim) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>>
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
      EXPECT_EQ(result[i][j], a0[i][j] * a1[0][j])
          << "result differ at pos (" << i << "," << j << "), expect "
          << a0[i][j] * a1[0][j] << " got " << result[i][j] << "\n";
    }
  }
}