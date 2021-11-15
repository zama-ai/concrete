#include "end_to_end_jit_test.h"

#define GET_3D(tensor, i, j, k, di, dj, dk) (tensor)[i * dj * dk + j * dk + k]
#define GET_2D(tensor, i, j, di, dj) (tensor)[i * dj + j]

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg add_eint_int////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, add_eint_int_term_to_term) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x!HLFHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!HLFHE.eint<6>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<6>>, tensor<4xi7>) -> tensor<4x!HLFHE.eint<6>>
    return %res : tensor<4x!HLFHE.eint<6>>
  }
)XXX");
  std::vector<uint8_t> a0{31, 6, 12, 9};
  std::vector<uint8_t> a1{32, 9, 2, 3};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*res)[i], a0[i] + a1[i]);
  }
}

// Same as add_eint_int_term_to_term test above, but returning a lambda argument
TEST(End2EndJit_HLFHELinalg, add_eint_int_term_to_term_ret_lambda_argument) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x!HLFHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!HLFHE.eint<6>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<6>>, tensor<4xi7>) -> tensor<4x!HLFHE.eint<6>>
    return %res : tensor<4x!HLFHE.eint<6>>
  }
)XXX");
  std::vector<uint8_t> a0{31, 6, 12, 9};
  std::vector<uint8_t> a1{32, 9, 2, 3};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::unique_ptr<mlir::zamalang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::zamalang::LambdaArgument>>(
          {&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  mlir::zamalang::TensorLambdaArgument<mlir::zamalang::IntLambdaArgument<>>
      &resp = (*res)
                  ->cast<mlir::zamalang::TensorLambdaArgument<
                      mlir::zamalang::IntLambdaArgument<>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)1);
  ASSERT_EQ(resp.getDimensions().at(0), 4);
  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 4);

  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(resp.getValue()[i], a0[i] + a1[i]);
  }
}

// Same as add_eint_int_term_to_term_ret_lambda_argument, but returning a
// multi-dimensional tensor
TEST(End2EndJit_HLFHELinalg,
     add_eint_int_term_to_term_ret_lambda_argument_multi_dim) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x2x3x!HLFHE.eint<6>>, %a1: tensor<4x2x3xi7>) -> tensor<4x2x3x!HLFHE.eint<6>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x2x3x!HLFHE.eint<6>>, tensor<4x2x3xi7>) -> tensor<4x2x3x!HLFHE.eint<6>>
    return %res : tensor<4x2x3x!HLFHE.eint<6>>
  }
)XXX");
  std::vector<uint8_t> a0{31, 6, 12, 9, 1, 2, 3, 4, 9, 0, 3, 2,
                          2,  1, 0,  6, 3, 6, 2, 8, 0, 0, 4, 3};
  std::vector<uint8_t> a1{32, 9, 2, 3, 6, 6, 2, 1, 1, 6, 9, 7,
                          3,  5, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(a0, {4, 2, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(a1, {4, 2, 3});

  llvm::Expected<std::unique_ptr<mlir::zamalang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::zamalang::LambdaArgument>>(
          {&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  mlir::zamalang::TensorLambdaArgument<mlir::zamalang::IntLambdaArgument<>>
      &resp = (*res)
                  ->cast<mlir::zamalang::TensorLambdaArgument<
                      mlir::zamalang::IntLambdaArgument<>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)3);
  ASSERT_EQ(resp.getDimensions().at(0), 4);
  ASSERT_EQ(resp.getDimensions().at(1), 2);
  ASSERT_EQ(resp.getDimensions().at(2), 3);
  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 4 * 3 * 2);

  for (size_t i = 0; i < 4 * 3 * 2; i++) {
    ASSERT_EQ(resp.getValue()[i], a0[i] + a1[i]);
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_int_term_to_term_broadcast) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x1x4x!HLFHE.eint<5>>, %a1: tensor<1x4x4xi6>) -> tensor<4x4x4x!HLFHE.eint<5>> {
    %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x1x4x!HLFHE.eint<5>>, tensor<1x4x4xi6>) -> tensor<4x4x4x!HLFHE.eint<5>>
    return %res : tensor<4x4x4x!HLFHE.eint<5>>
  }
)XXX");
  uint8_t a0[4][1][4]{
      {{1, 2, 3, 4}},
      {{5, 6, 7, 8}},
      {{9, 10, 11, 12}},
      {{13, 14, 15, 16}},
  };
  uint8_t a1[1][4][4]{
      {
          {1, 2, 3, 4},
          {5, 6, 7, 8},
          {9, 10, 11, 12},
          {13, 14, 15, 16},
      },
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::MutableArrayRef<uint8_t>((uint8_t *)a0, 4 * 1 * 4), {4, 1, 4});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::MutableArrayRef<uint8_t>((uint8_t *)a1, 1 * 4 * 4), {1, 4, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 4 * 4 * 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ((*res)[i * 16 + j * 4 + k], a0[i][0][k] + a1[0][j][k])
            << "result differ at pos " << i << "," << j << "," << k;
      }
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_int_matrix_column) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
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
)XXX");
  uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  uint8_t a1[3][1]{
      {1},
      {2},
      {3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::MutableArrayRef<uint8_t>((uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::MutableArrayRef<uint8_t>((uint8_t *)a0, 3 * 1), {3, 1});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] + a1[i][0]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_int_matrix_line) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
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
)XXX");
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {1, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] + a1[0][j]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_int_matrix_line_missing_dim) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
   // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
   func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>> {
     %res = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>>
     return %res : tensor<3x3x!HLFHE.eint<4>>
   }
 )XXX");
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] + a1[0][j]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg add_eint ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, add_eint_term_to_term) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x!HLFHE.eint<6>>, %a1: tensor<4x!HLFHE.eint<6>>) -> tensor<4x!HLFHE.eint<6>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<4x!HLFHE.eint<6>>, tensor<4x!HLFHE.eint<6>>) -> tensor<4x!HLFHE.eint<6>>
    return %res : tensor<4x!HLFHE.eint<6>>
  }
)XXX");

  std::vector<uint8_t> a0{31, 6, 12, 9};
  std::vector<uint8_t> a1{32, 9, 2, 3};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*res)[i], a0[i] + a1[i])
        << "result differ at pos " << i << ", expect " << a0[i] + a1[i]
        << " got " << (*res)[i];
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_term_to_term_broadcast) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x1x4x!HLFHE.eint<5>>, %a1:
  tensor<1x4x4x!HLFHE.eint<5>>) -> tensor<4x4x4x!HLFHE.eint<5>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) :
    (tensor<4x1x4x!HLFHE.eint<5>>, tensor<1x4x4x!HLFHE.eint<5>>) ->
    tensor<4x4x4x!HLFHE.eint<5>> return %res : tensor<4x4x4x!HLFHE.eint<5>>
  }
)XXX");
  uint8_t a0[4][1][4]{
      {{1, 2, 3, 4}},
      {{5, 6, 7, 8}},
      {{9, 10, 11, 12}},
      {{13, 14, 15, 16}},
  };
  uint8_t a1[1][4][4]{
      {
          {1, 2, 3, 4},
          {5, 6, 7, 8},
          {9, 10, 11, 12},
          {13, 14, 15, 16},
      },
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::MutableArrayRef<uint8_t>((uint8_t *)a0, 4 * 1 * 4), {4, 1, 4});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::MutableArrayRef<uint8_t>((uint8_t *)a1, 1 * 4 * 4), {1, 4, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 4 * 4 * 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ((*res)[i * 16 + j * 4 + k], a0[i][0][k] + a1[0][j][k])
            << "result differ at pos " << i << "," << j << "," << k;
      }
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_matrix_column) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
  //
  // [1,2,3]   [1]   [2,3,4]
  // [4,5,6] + [2] = [6,7,8]
  // [7,8,9]   [3]   [10,11,12]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1:
  tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
    return %res : tensor<3x3x!HLFHE.eint<4>>
  }
)XXX");
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

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3, 1});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] + a1[i][0]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_matrix_line) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
  //
  // [1,2,3]             [2,4,6]
  // [4,5,6] + [1,2,3] = [5,7,9]
  // [7,8,9]             [8,10,12]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1:
  tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>,
    tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> return %res :
    tensor<3x3x!HLFHE.eint<4>>
  }
)XXX");
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {1, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] + a1[0][j]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_matrix_line_missing_dim) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
    return %res : tensor<3x3x!HLFHE.eint<4>>
  }
)XXX");
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] + a1[0][j]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, add_eint_tensor_dim_equals_1) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Broadcasting shouldn't happen when some dimensions are equals to 1
  func @main(%arg0: tensor<3x1x2x!HLFHE.eint<5>>, %arg1: tensor<3x1x2x!HLFHE.eint<5>>) -> tensor<3x1x2x!HLFHE.eint<5>> {
    %1 = "HLFHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x1x2x!HLFHE.eint<5>>, tensor<3x1x2x!HLFHE.eint<5>>) -> tensor<3x1x2x!HLFHE.eint<5>>
    return %1 : tensor<3x1x2x!HLFHE.eint<5>>
  }
)XXX");
  const uint8_t a0[3][1][2]{
      {{1, 2}},
      {{4, 5}},
      {{7, 8}},
  };
  const uint8_t a1[3][1][2]{
      {{8, 10}},
      {{12, 14}},
      {{16, 18}},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 2), {3, 1, 2});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a1, 3 * 2), {3, 1, 2});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 1 * 2);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 1; j++) {
      for (size_t k = 0; k < 2; k++) {
        EXPECT_EQ((*res)[i * 2 + j + k], a0[i][j][k] + a1[i][j][k]);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg sub_int_eint ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, sub_int_eint_term_to_term) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term substraction of `%a0` with `%a1`
  func @main(%a0: tensor<4xi5>, %a1: tensor<4x!HLFHE.eint<4>>) -> tensor<4x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4xi5>, tensor<4x!HLFHE.eint<4>>) -> tensor<4x!HLFHE.eint<4>>
    return %res : tensor<4x!HLFHE.eint<4>>
  }
)XXX");
  std::vector<uint8_t> a0{32, 9, 12, 9};
  std::vector<uint8_t> a1{31, 6, 2, 3};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*res)[i], a0[i] - a1[i]);
  }
}

TEST(End2EndJit_HLFHELinalg, sub_int_eint_term_to_term_broadcast) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term substraction of `%a0` with `%a1`, where dimensions equals to one are stretched.
  func @main(%a0: tensor<4x1x4xi8>, %a1: tensor<1x4x4x!HLFHE.eint<7>>) -> tensor<4x4x4x!HLFHE.eint<7>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4x1x4xi8>, tensor<1x4x4x!HLFHE.eint<7>>) -> tensor<4x4x4x!HLFHE.eint<7>>
    return %res : tensor<4x4x4x!HLFHE.eint<7>>
  }
)XXX");
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
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 4 * 1 * 4), {4, 1, 4});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a1, 1 * 4 * 4), {1, 4, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 4 * 4 * 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        uint8_t expected = a0[i][0][k] - a1[0][j][k];
        EXPECT_EQ((*res)[i * 16 + j * 4 + k], expected);
      }
    }
  }
}

TEST(End2EndJit_HLFHELinalg, sub_int_eint_matrix_column) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the substraction of a 3x3 matrix of integers and a 3x1 matrix (a column) of encrypted integers.
  //
  // [1,2,3]   [1]   [0,2,3]
  // [4,5,6] - [2] = [2,3,4]
  // [7,8,9]   [3]   [4,5,6]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<3x1x!HLFHE.eint<4>>) ->
  tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>,
    tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> return %res :
    tensor<3x3x!HLFHE.eint<4>>
  }
)XXX");
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

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3, 1});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] - a1[i][0]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, sub_int_eint_matrix_line) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the substraction of a 3x3 matrix of integers and a 1x3 matrix (a line) of encrypted integers.
  //
  // [1,2,3]             [0,0,0]
  // [4,5,6] + [1,2,3] = [3,3,3]
  // [7,8,9]             [6,6,6]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<1x3x!HLFHE.eint<4>>) ->
  tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>,
    tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> return %res :
    tensor<3x3x!HLFHE.eint<4>>
  }
)XXX");
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {1, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] - a1[0][j]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, sub_int_eint_matrix_line_missing_dim) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
    return %res : tensor<3x3x!HLFHE.eint<4>>
  }
)XXX");
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] - a1[0][j]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg mul_eint_int ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, mul_eint_int_term_to_term) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term multiplication of `%a0` with `%a1`
  func @main(%a0: tensor<4x!HLFHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!HLFHE.eint<6>> {
    %res = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<6>>, tensor<4xi7>) -> tensor<4x!HLFHE.eint<6>>
    return %res : tensor<4x!HLFHE.eint<6>>
  }
)XXX");
  std::vector<uint8_t> a0{31, 6, 12, 9};
  std::vector<uint8_t> a1{2, 3, 2, 3};

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*res)[i], a0[i] * a1[i]);
  }
}

TEST(End2EndJit_HLFHELinalg, mul_eint_int_term_to_term_broadcast) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term multiplication of `%a0` with `%a1`, where dimensions equals to one are stretched.
  func @main(%a0: tensor<4x1x4x!HLFHE.eint<6>>, %a1: tensor<1x4x4xi7>) -> tensor<4x4x4x!HLFHE.eint<6>> {
    %res = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x1x4x!HLFHE.eint<6>>, tensor<1x4x4xi7>) -> tensor<4x4x4x!HLFHE.eint<6>>
    return %res : tensor<4x4x4x!HLFHE.eint<6>>
  }
)XXX");
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

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 4 * 1 * 4), {4, 1, 4});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a1, 1 * 4 * 4), {1, 4, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 4 * 4 * 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        uint8_t expected = a0[i][0][k] * a1[0][j][k];
        EXPECT_EQ((*res)[i * 16 + j * 4 + k], expected);
      }
    }
  }
}

TEST(End2EndJit_HLFHELinalg, mul_eint_int_matrix_column) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
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
)XXX");
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

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3, 1});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] * a1[i][0]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, mul_eint_int_matrix_line) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
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
)XXX");
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {1, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] * a1[0][j]);
    }
  }
}

TEST(End2EndJit_HLFHELinalg, mul_eint_int_matrix_line_missing_dim) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3x!HLFHE.eint<4>>, %a1: tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>> {
    %res = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>>
    return %res : tensor<3x3x!HLFHE.eint<4>>
  }
)XXX");
  const uint8_t a0[3][3]{
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };
  const uint8_t a1[1][3]{
      {1, 2, 3},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], a0[i][j] * a1[0][j]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg apply_lookup_table /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, apply_lookup_table) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the lookup of 3x3 matrix of encrypted indices of with 2 on a table of size 4=2² of clear integers.
    //
    // [0,1,2]                 [1,3,5]
    // [3,0,1] lut [1,3,5,7] = [7,1,3]
    // [2,3,0]                 [5,7,1]
    func @main(%t: tensor<3x3x!HLFHE.eint<2>>) -> tensor<3x3x!HLFHE.eint<3>> {
      %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
      %res = "HLFHELinalg.apply_lookup_table"(%t, %lut) : (tensor<3x3x!HLFHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!HLFHE.eint<3>>
      return %res : tensor<3x3x!HLFHE.eint<3>>
    }
)XXX");
  const uint8_t t[3][3]{
      {0, 1, 2},
      {3, 0, 1},
      {2, 3, 0},
  };
  const uint8_t expected[3][3]{
      {1, 3, 5},
      {7, 1, 3},
      {5, 7, 1},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      tArg(llvm::ArrayRef<uint8_t>((const uint8_t *)t, 3 * 3), {3, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&tArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg apply_multi_lookup_table
// /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, apply_multi_lookup_table) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the lookup of 3x3 matrix of encrypted indices of width 2 on a 3x3 matrix of tables of size 4=2² of clear integers.
    func @main(%arg0: tensor<3x3x!HLFHE.eint<2>>, %arg1: tensor<3x3x4xi64>) -> tensor<3x3x!HLFHE.eint<2>> {
      %1 = "HLFHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<3x3x!HLFHE.eint<2>>, tensor<3x3x4xi64>) -> tensor<3x3x!HLFHE.eint<2>>
      return %1: tensor<3x3x!HLFHE.eint<2>>
    }
)XXX");
  const uint8_t t[3][3]{
      {0, 1, 2},
      {3, 0, 1},
      {2, 3, 0},
  };
  const uint64_t luts[3][3][4]{
      {{1, 3, 5, 7}, {0, 4, 1, 3}, {3, 2, 5, 0}},
      {{0, 2, 1, 2}, {7, 1, 0, 2}, {0, 1, 2, 3}},
      {{2, 1, 0, 3}, {0, 1, 2, 3}, {6, 5, 4, 3}},
  };
  const uint8_t expected[3][3]{
      {1, 4, 5},
      {2, 7, 1},
      {0, 3, 6},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      tArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)t, 3 * 3), {3, 3});

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint64_t>>
      lutsArg(llvm::MutableArrayRef<uint64_t>((uint64_t *)luts, 3 * 3 * 4),
              {3, 3, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&tArg, &lutsArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_HLFHELinalg, apply_multi_lookup_table_with_boradcast) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the lookup of 3x3 matrix of encrypted indices of width 2 on a vector of 3 tables of size 4=2² of clear integers.
    func @main(%arg0: tensor<3x3x!HLFHE.eint<2>>, %arg1: tensor<3x4xi64>) -> tensor<3x3x!HLFHE.eint<2>> {
      %1 = "HLFHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<3x3x!HLFHE.eint<2>>, tensor<3x4xi64>) -> tensor<3x3x!HLFHE.eint<2>>
      return %1: tensor<3x3x!HLFHE.eint<2>>
    }
)XXX");
  const uint8_t t[3][3]{
      {0, 1, 2},
      {3, 0, 1},
      {2, 3, 0},
  };
  const uint64_t luts[3][4]{
      {1, 3, 5, 7},
      {0, 2, 1, 3},
      {2, 1, 0, 6},
  };
  const uint8_t expected[3][3]{
      {1, 2, 0},
      {7, 0, 1},
      {5, 3, 2},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      tArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)t, 3 * 3), {3, 3});

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint64_t>>
      lutsArg(llvm::MutableArrayRef<uint64_t>((uint64_t *)luts, 3 * 4),
              {3, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&tArg, &lutsArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg dot_eint_int ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(CompileAndRunTensorEncrypted, dot_eint_int_7) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: tensor<4x!HLFHE.eint<7>>,
                   %arg1: tensor<4xi8>) -> !HLFHE.eint<7>
{
  %ret = "HLFHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<4x!HLFHE.eint<7>>, tensor<4xi8>) -> !HLFHE.eint<7>
  return %ret : !HLFHE.eint<7>
}
)XXX");
  static uint8_t arg0[] = {0, 1, 2, 3};
  static uint8_t arg1[] = {0, 1, 2, 3};

  llvm::Expected<uint64_t> res =
      lambda(arg0, ARRAY_SIZE(arg0), arg1, ARRAY_SIZE(arg1));

  ASSERT_EXPECTED_VALUE(res, 14);
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg neg_eint /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, neg_eint) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the negation of a 3x3 matrix of encrypted integers of width 2.
    //
    //        ([0,1,2])   [0,7,6]
    // negate ([3,4,5]) = [5,4,3]
    //        ([6,7,0])   [2,1,0]
    func @main(%t: tensor<3x3x!HLFHE.eint<2>>) -> tensor<3x3x!HLFHE.eint<2>> {
      %res = "HLFHELinalg.neg_eint"(%t) : (tensor<3x3x!HLFHE.eint<2>>) -> tensor<3x3x!HLFHE.eint<2>>
      return %res : tensor<3x3x!HLFHE.eint<2>>
    }
)XXX");
  const uint8_t t[3][3]{
      {0, 1, 2},
      {3, 4, 5},
      {6, 7, 0},
  };
  const uint8_t expected[3][3]{
      {0, 7, 6},
      {5, 4, 3},
      {2, 1, 0},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      tArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)t, 3 * 3), {3, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&tArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// HLFHELinalg matmul_eint_int ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_HLFHELinalg, matmul_eint_int) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the matrix multiplication of a 3x2 matrix of encrypted integers and a 2x3 matrix of integers.
  //         [ 1, 2, 3]
  //         [ 2, 3, 4]
  //       *
  // [1,2]   [ 5, 8,11]
  // [3,4] = [11,18,25]
  // [5,6]   [17,28,39]
  func @main(%a: tensor<3x2x!HLFHE.eint<6>>, %b: tensor<2x3xi7>) -> tensor<3x3x!HLFHE.eint<6>> {
    %0 = "HLFHELinalg.matmul_eint_int"(%a, %b) : (tensor<3x2x!HLFHE.eint<6>>, tensor<2x3xi7>) -> tensor<3x3x!HLFHE.eint<6>>
    return %0 : tensor<3x3x!HLFHE.eint<6>>
  }
)XXX");
  const uint8_t A[3][2]{
      {1, 2},
      {3, 4},
      {5, 6},
  };
  const uint8_t B[2][3]{
      {1, 2, 3},
      {2, 3, 4},
  };
  const uint8_t expected[3][3]{
      {5, 8, 11},
      {11, 18, 25},
      {17, 28, 39},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      aArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)A, 3 * 2), {3, 2});
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      bArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)B, 2 * 3), {2, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&aArg, &bArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// linalg.tensor_collapse_shape ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_Linalg, tensor_collapse_shape) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%a: tensor<2x2x4x!HLFHE.eint<6>>) -> tensor<2x8x!HLFHE.eint<6>> {
  %0 = linalg.tensor_collapse_shape %a [[0],[1,2]]  : tensor<2x2x4x!HLFHE.eint<6>> into tensor<2x8x!HLFHE.eint<6>>
  return %0 : tensor<2x8x!HLFHE.eint<6>>
}
)XXX");
  static uint8_t A[2][2][4]{
      {{1, 2, 3, 4}, {5, 6, 7, 8}},
      {{10, 11, 12, 13}, {14, 15, 16, 17}},
  };
  static uint8_t expected[2][8]{
      {1, 2, 3, 4, 5, 6, 7, 8},
      {10, 11, 12, 13, 14, 15, 16, 17},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      aArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)A, 2 * 2 * 4), {2, 2, 4});

  llvm::Expected<std::unique_ptr<mlir::zamalang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::zamalang::LambdaArgument>>(
          {&aArg});

  ASSERT_EXPECTED_SUCCESS(res);

  mlir::zamalang::TensorLambdaArgument<mlir::zamalang::IntLambdaArgument<>>
      &resp = (*res)
                  ->cast<mlir::zamalang::TensorLambdaArgument<
                      mlir::zamalang::IntLambdaArgument<>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)2);
  ASSERT_EQ(resp.getDimensions().at(0), 2);
  ASSERT_EQ(resp.getDimensions().at(1), 8);
  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 2 * 8);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 8; j++) {
      EXPECT_EQ(resp.getValue()[i * 8 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// linalg.tensor_expand_shape ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_Linalg, tensor_expand_shape) {

  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%a: tensor<2x8x!HLFHE.eint<6>>) -> tensor<2x2x4x!HLFHE.eint<6>> {
  %0 = linalg.tensor_expand_shape %a [[0],[1,2]]  : tensor<2x8x!HLFHE.eint<6>> into tensor<2x2x4x!HLFHE.eint<6>>
  return %0 : tensor<2x2x4x!HLFHE.eint<6>>
}
)XXX");

  static uint8_t A[2][8]{
      {1, 2, 3, 4, 5, 6, 7, 8},
      {10, 11, 12, 13, 14, 15, 16, 17},
  };
  static uint8_t expected[2][2][4]{
      {{1, 2, 3, 4}, {5, 6, 7, 8}},
      {{10, 11, 12, 13}, {14, 15, 16, 17}},
  };

  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint8_t>>
      aArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)A, 2 * 8), {2, 8});

  llvm::Expected<std::unique_ptr<mlir::zamalang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::zamalang::LambdaArgument>>(
          {&aArg});

  ASSERT_EXPECTED_SUCCESS(res);

  mlir::zamalang::TensorLambdaArgument<mlir::zamalang::IntLambdaArgument<>>
      &resp = (*res)
                  ->cast<mlir::zamalang::TensorLambdaArgument<
                      mlir::zamalang::IntLambdaArgument<>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)3);
  ASSERT_EQ(resp.getDimensions().at(0), 2);
  ASSERT_EQ(resp.getDimensions().at(1), 2);
  ASSERT_EQ(resp.getDimensions().at(2), 4);
  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 2 * 2 * 4);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ(resp.getValue()[i * 8 + j * 4 + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}