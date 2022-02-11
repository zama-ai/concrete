#include "end_to_end_jit_test.h"

namespace Z = mlir::concretelang;
template <class Elmt>
using tensorArgTy = Z::TensorLambdaArgument<Z::IntLambdaArgument<Elmt>>;

#define GET_3D(tensor, i, j, k, di, dj, dk) (tensor)[i * dj * dk + j * dk + k]
#define GET_2D(tensor, i, j, di, dj) (tensor)[i * dj + j]

///////////////////////////////////////////////////////////////////////////////
// FHELinalg add_eint_int////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, add_eint_int_term_to_term) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!FHE.eint<6>> {
    %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4xi7>) -> tensor<4x!FHE.eint<6>>
    return %res : tensor<4x!FHE.eint<6>>
  }
)XXX");
  std::vector<uint8_t> a0{31, 6, 12, 9};
  std::vector<uint8_t> a1{32, 9, 2, 3};

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*res)[i], (uint64_t)a0[i] + a1[i]);
  }
}

// Same as add_eint_int_term_to_term test above, but returning a lambda argument
TEST(End2EndJit_FHELinalg, add_eint_int_term_to_term_ret_lambda_argument) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!FHE.eint<6>> {
    %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4xi7>) -> tensor<4x!FHE.eint<6>>
    return %res : tensor<4x!FHE.eint<6>>
  }
)XXX");
  std::vector<uint8_t> a0{31, 6, 12, 9};
  std::vector<uint8_t> a1{32, 9, 2, 3};

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
          {&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<>> &resp =
      (*res)
          ->cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)1);
  ASSERT_EQ(resp.getDimensions().at(0), 4);
  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 4);

  for (size_t i = 0; i < 4; i++) {
    ASSERT_EQ(resp.getValue()[i], (uint64_t)a0[i] + a1[i]);
  }
}

// Same as add_eint_int_term_to_term_ret_lambda_argument, but returning a
// multi-dimensional tensor
TEST(End2EndJit_FHELinalg,
     add_eint_int_term_to_term_ret_lambda_argument_multi_dim) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x2x3x!FHE.eint<6>>, %a1: tensor<4x2x3xi7>) -> tensor<4x2x3x!FHE.eint<6>> {
    %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x2x3x!FHE.eint<6>>, tensor<4x2x3xi7>) -> tensor<4x2x3x!FHE.eint<6>>
    return %res : tensor<4x2x3x!FHE.eint<6>>
  }
)XXX");
  std::vector<uint8_t> a0{31, 6, 12, 9, 1, 2, 3, 4, 9, 0, 3, 2,
                          2,  1, 0,  6, 3, 6, 2, 8, 0, 0, 4, 3};
  std::vector<uint8_t> a1{32, 9, 2, 3, 6, 6, 2, 1, 1, 6, 9, 7,
                          3,  5, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(a0, {4, 2, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(a1, {4, 2, 3});

  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
          {&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<>> &resp =
      (*res)
          ->cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(resp.getDimensions().size(), (size_t)3);
  ASSERT_EQ(resp.getDimensions().at(0), 4);
  ASSERT_EQ(resp.getDimensions().at(1), 2);
  ASSERT_EQ(resp.getDimensions().at(2), 3);
  ASSERT_EXPECTED_VALUE(resp.getNumElements(), 4 * 3 * 2);

  for (size_t i = 0; i < 4 * 3 * 2; i++) {
    ASSERT_EQ(resp.getValue()[i], (uint64_t)a0[i] + a1[i]);
  }
}

TEST(End2EndJit_FHELinalg, add_eint_int_term_to_term_broadcast) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x1x4x!FHE.eint<5>>, %a1: tensor<1x4x4xi6>) -> tensor<4x4x4x!FHE.eint<5>> {
    %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<5>>, tensor<1x4x4xi6>) -> tensor<4x4x4x!FHE.eint<5>>
    return %res : tensor<4x4x4x!FHE.eint<5>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::MutableArrayRef<uint8_t>((uint8_t *)a0, 4 * 1 * 4), {4, 1, 4});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::MutableArrayRef<uint8_t>((uint8_t *)a1, 1 * 4 * 4), {1, 4, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (size_t)4 * 4 * 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ((*res)[i * 16 + j * 4 + k],
                  (uint64_t)a0[i][0][k] + a1[0][j][k])
            << "result differ at pos " << i << "," << j << "," << k;
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, add_eint_int_matrix_column) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
  //
  // [1,2,3]   [1]   [2,3,4]
  // [4,5,6] + [2] = [6,7,8]
  // [7,8,9]   [3]   [10,11,12]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1: tensor<3x1xi5>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!FHE.eint<4>>
    return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::MutableArrayRef<uint8_t>((uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::MutableArrayRef<uint8_t>((uint8_t *)a0, 3 * 1), {3, 1});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] + a1[i][0]);
    }
  }
}

TEST(End2EndJit_FHELinalg, add_eint_int_matrix_line) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
  //
  // [1,2,3]             [2,4,6]
  // [4,5,6] + [1,2,3] = [5,7,9]
  // [7,8,9]             [8,10,12]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1: tensor<1x3xi5>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!FHE.eint<4>>
    return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {1, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] + a1[0][j]);
    }
  }
}

TEST(End2EndJit_FHELinalg, add_eint_int_matrix_line_missing_dim) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
   // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
   func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1: tensor<3xi5>) -> tensor<3x3x!FHE.eint<4>> {
     %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!FHE.eint<4>>
     return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] + a1[0][j]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// FHELinalg add_eint ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, add_eint_term_to_term) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4x!FHE.eint<6>>) -> tensor<4x!FHE.eint<6>> {
    %res = "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4x!FHE.eint<6>>) -> tensor<4x!FHE.eint<6>>
    return %res : tensor<4x!FHE.eint<6>>
  }
)XXX");

  std::vector<uint8_t> a0{31, 6, 12, 9};
  std::vector<uint8_t> a1{32, 9, 2, 3};

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*res)[i], (uint64_t)a0[i] + a1[i])
        << "result differ at pos " << i << ", expect " << a0[i] + a1[i]
        << " got " << (*res)[i];
  }
}

TEST(End2EndJit_FHELinalg, add_eint_term_to_term_broadcast) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term addition of `%a0` with `%a1`
  func @main(%a0: tensor<4x1x4x!FHE.eint<5>>, %a1:
  tensor<1x4x4x!FHE.eint<5>>) -> tensor<4x4x4x!FHE.eint<5>> {
    %res = "FHELinalg.add_eint"(%a0, %a1) :
    (tensor<4x1x4x!FHE.eint<5>>, tensor<1x4x4x!FHE.eint<5>>) ->
    tensor<4x4x4x!FHE.eint<5>> return %res : tensor<4x4x4x!FHE.eint<5>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::MutableArrayRef<uint8_t>((uint8_t *)a0, 4 * 1 * 4), {4, 1, 4});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::MutableArrayRef<uint8_t>((uint8_t *)a1, 1 * 4 * 4), {1, 4, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)4 * 4 * 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ((*res)[i * 16 + j * 4 + k],
                  (uint64_t)a0[i][0][k] + a1[0][j][k])
            << "result differ at pos " << i << "," << j << "," << k;
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, add_eint_matrix_column) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
  //
  // [1,2,3]   [1]   [2,3,4]
  // [4,5,6] + [2] = [6,7,8]
  // [7,8,9]   [3]   [10,11,12]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1:
  tensor<3x1x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>
    return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3, 1});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] + a1[i][0]);
    }
  }
}

TEST(End2EndJit_FHELinalg, add_eint_matrix_line) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
  //
  // [1,2,3]             [2,4,6]
  // [4,5,6] + [1,2,3] = [5,7,9]
  // [7,8,9]             [8,10,12]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1:
  tensor<1x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>,
    tensor<1x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>> return %res :
    tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {1, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] + a1[0][j]);
    }
  }
}

TEST(End2EndJit_FHELinalg, add_eint_matrix_line_missing_dim) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1: tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>
    return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] + a1[0][j]);
    }
  }
}

TEST(End2EndJit_FHELinalg, add_eint_tensor_dim_equals_1) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Broadcasting shouldn't happen when some dimensions are equals to 1
  func @main(%arg0: tensor<3x1x2x!FHE.eint<5>>, %arg1: tensor<3x1x2x!FHE.eint<5>>) -> tensor<3x1x2x!FHE.eint<5>> {
    %1 = "FHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x1x2x!FHE.eint<5>>, tensor<3x1x2x!FHE.eint<5>>) -> tensor<3x1x2x!FHE.eint<5>>
    return %1 : tensor<3x1x2x!FHE.eint<5>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 2), {3, 1, 2});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a1, 3 * 2), {3, 1, 2});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 1 * 2);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 1; j++) {
      for (size_t k = 0; k < 2; k++) {
        EXPECT_EQ((*res)[i * 2 + j + k], (uint64_t)a0[i][j][k] + a1[i][j][k]);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// FHELinalg sub_int_eint ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, sub_int_eint_term_to_term) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term substraction of `%a0` with `%a1`
  func @main(%a0: tensor<4xi5>, %a1: tensor<4x!FHE.eint<4>>) -> tensor<4x!FHE.eint<4>> {
    %res = "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4xi5>, tensor<4x!FHE.eint<4>>) -> tensor<4x!FHE.eint<4>>
    return %res : tensor<4x!FHE.eint<4>>
  }
)XXX");
  std::vector<uint8_t> a0{32, 9, 12, 9};
  std::vector<uint8_t> a1{31, 6, 2, 3};

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*res)[i], (uint64_t)a0[i] - a1[i]);
  }
}

TEST(End2EndJit_FHELinalg, sub_int_eint_term_to_term_broadcast) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term substraction of `%a0` with `%a1`, where dimensions equals to one are stretched.
  func @main(%a0: tensor<4x1x4xi8>, %a1: tensor<1x4x4x!FHE.eint<7>>) -> tensor<4x4x4x!FHE.eint<7>> {
    %res = "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4x1x4xi8>, tensor<1x4x4x!FHE.eint<7>>) -> tensor<4x4x4x!FHE.eint<7>>
    return %res : tensor<4x4x4x!FHE.eint<7>>
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
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 4 * 1 * 4), {4, 1, 4});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a1, 1 * 4 * 4), {1, 4, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)4 * 4 * 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        uint8_t expected = a0[i][0][k] - a1[0][j][k];
        EXPECT_EQ((*res)[i * 16 + j * 4 + k], (uint64_t)expected);
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, sub_int_eint_matrix_column) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the substraction of a 3x3 matrix of integers and a 3x1 matrix (a column) of encrypted integers.
  //
  // [1,2,3]   [1]   [0,2,3]
  // [4,5,6] - [2] = [2,3,4]
  // [7,8,9]   [3]   [4,5,6]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<3x1x!FHE.eint<4>>) ->
  tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>,
    tensor<3x1x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>> return %res :
    tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3, 1});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] - a1[i][0]);
    }
  }
}

TEST(End2EndJit_FHELinalg, sub_int_eint_matrix_line) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the substraction of a 3x3 matrix of integers and a 1x3 matrix (a line) of encrypted integers.
  //
  // [1,2,3]             [0,0,0]
  // [4,5,6] + [1,2,3] = [3,3,3]
  // [7,8,9]             [6,6,6]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<1x3x!FHE.eint<4>>) ->
  tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>,
    tensor<1x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>> return %res :
    tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {1, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] - a1[0][j]);
    }
  }
}

TEST(End2EndJit_FHELinalg, sub_int_eint_matrix_line_missing_dim) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3xi5>, %a1: tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>
    return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] - a1[0][j]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// FHELinalg mul_eint_int ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, mul_eint_int_term_to_term) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term multiplication of `%a0` with `%a1`
  func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!FHE.eint<6>> {
    %res = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4xi7>) -> tensor<4x!FHE.eint<6>>
    return %res : tensor<4x!FHE.eint<6>>
  }
)XXX");
  std::vector<uint8_t> a0{31, 6, 12, 9};
  std::vector<uint8_t> a1{2, 3, 2, 3};

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(a0);
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(a1);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*res)[i], (uint64_t)a0[i] * a1[i]);
  }
}

TEST(End2EndJit_FHELinalg, mul_eint_int_term_to_term_broadcast) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the term to term multiplication of `%a0` with `%a1`, where dimensions equals to one are stretched.
  func @main(%a0: tensor<4x1x4x!FHE.eint<6>>, %a1: tensor<1x4x4xi7>) -> tensor<4x4x4x!FHE.eint<6>> {
    %res = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<6>>, tensor<1x4x4xi7>) -> tensor<4x4x4x!FHE.eint<6>>
    return %res : tensor<4x4x4x!FHE.eint<6>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 4 * 1 * 4), {4, 1, 4});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a1, 1 * 4 * 4), {1, 4, 4});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)4 * 4 * 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        uint8_t expected = a0[i][0][k] * a1[0][j][k];
        EXPECT_EQ((*res)[i * 16 + j * 4 + k], (uint64_t)expected);
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, mul_eint_int_matrix_column) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the multiplication of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of integers.
  //
  // [1,2,3]   [1]   [1,2,3]
  // [4,5,6] * [2] = [8,10,18]
  // [7,8,9]   [3]   [21,24,27]
  //
  // The dimension #1 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1: tensor<3x1xi5>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!FHE.eint<4>>
    return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3, 1});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] * a1[i][0]);
    }
  }
}

TEST(End2EndJit_FHELinalg, mul_eint_int_matrix_line) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the multiplication of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
  //
  // [1,2,3]             [2,4,6]
  // [4,5,6] * [1,2,3] = [5,7,9]
  // [7,8,9]             [8,10,12]
  //
  // The dimension #2 of operand #2 is stretched as it is equals to 1.
  func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1: tensor<1x3xi5>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!FHE.eint<4>>
    return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {1, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] * a1[0][j]);
    }
  }
}

TEST(End2EndJit_FHELinalg, mul_eint_int_matrix_line_missing_dim) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
  func @main(%a0: tensor<3x3x!FHE.eint<4>>, %a1: tensor<3xi5>) -> tensor<3x3x!FHE.eint<4>> {
    %res = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!FHE.eint<4>>
    return %res : tensor<3x3x!FHE.eint<4>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg0(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 3), {3, 3});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg1(llvm::ArrayRef<uint8_t>((const uint8_t *)a0, 3 * 1), {3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&arg0, &arg1});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], (uint64_t)a0[i][j] * a1[0][j]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// FHELinalg apply_lookup_table /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, apply_lookup_table) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the lookup of 3x3 matrix of encrypted indices of with 2 on a table of size 4=2² of clear integers.
    //
    // [0,1,2]                 [1,3,5]
    // [3,0,1] lut [1,3,5,7] = [7,1,3]
    // [2,3,0]                 [5,7,1]
    func @main(%t: tensor<3x3x!FHE.eint<2>>) -> tensor<3x3x!FHE.eint<3>> {
      %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
      %res = "FHELinalg.apply_lookup_table"(%t, %lut) : (tensor<3x3x!FHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!FHE.eint<3>>
      return %res : tensor<3x3x!FHE.eint<3>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      tArg(llvm::ArrayRef<uint8_t>((const uint8_t *)t, 3 * 3), {3, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&tArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// FHELinalg apply_multi_lookup_table /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, apply_multi_lookup_table) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the lookup of 3x3 matrix of encrypted indices of width 2 on a 3x3 matrix of tables of size 4=2² of clear integers.
    func @main(%arg0: tensor<3x3x!FHE.eint<2>>, %arg1: tensor<3x3x4xi64>) -> tensor<3x3x!FHE.eint<2>> {
      %1 = "FHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<3x3x!FHE.eint<2>>, tensor<3x3x4xi64>) -> tensor<3x3x!FHE.eint<2>>
      return %1: tensor<3x3x!FHE.eint<2>>
    }
)XXX");
  uint8_t t[3][3]{
      {0, 1, 2},
      {3, 0, 1},
      {2, 3, 0},
  };
  uint64_t luts[3][3][4]{
      {{1, 3, 5, 7}, {0, 4, 1, 3}, {3, 2, 5, 0}},
      {{0, 2, 1, 2}, {7, 1, 0, 2}, {0, 1, 2, 3}},
      {{2, 1, 0, 3}, {0, 1, 2, 3}, {6, 5, 4, 3}},
  };
  const uint8_t expected[3][3]{
      {1, 4, 5},
      {2, 7, 1},
      {0, 3, 6},
  };

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      tArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)t, 3 * 3), {3, 3});

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>
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

TEST(End2EndJit_FHELinalg, apply_multi_lookup_table_with_boradcast) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the lookup of 3x3 matrix of encrypted indices of width 2 on a vector of 3 tables of size 4=2² of clear integers.
    func @main(%arg0: tensor<3x3x!FHE.eint<2>>, %arg1: tensor<3x4xi64>) -> tensor<3x3x!FHE.eint<2>> {
      %1 = "FHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<3x3x!FHE.eint<2>>, tensor<3x4xi64>) -> tensor<3x3x!FHE.eint<2>>
      return %1: tensor<3x3x!FHE.eint<2>>
    }
)XXX");
  uint8_t t[3][3]{
      {0, 1, 2},
      {3, 0, 1},
      {2, 3, 0},
  };
  uint64_t luts[3][4]{
      {1, 3, 5, 7},
      {0, 2, 1, 3},
      {2, 1, 0, 6},
  };
  const uint8_t expected[3][3]{
      {1, 2, 0},
      {7, 0, 1},
      {5, 3, 2},
  };

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      tArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)t, 3 * 3), {3, 3});

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>
      lutsArg(llvm::MutableArrayRef<uint64_t>((uint64_t *)luts, 3 * 4), {3, 4});

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
// FHELinalg apply_mapped_lookup_table ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, apply_mapped_lookup_table_sequential) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the lookup of 3x3 matrix of encrypted indices of width 2 of a 3x3 matrix of tables of size 4=2² of clear integers.
    func @main(%t: tensor<3x3x!FHE.eint<2>>, %luts: tensor<9x4xi64>, %map: tensor<3x3xindex>) -> tensor<3x3x!FHE.eint<2>> {
      %1 = "FHELinalg.apply_mapped_lookup_table"(%t, %luts, %map) :
        (tensor<3x3x!FHE.eint<2>>, tensor<9x4xi64>, tensor<3x3xindex>) -> tensor<3x3x!FHE.eint<2>>
      return %1: tensor<3x3x!FHE.eint<2>>
    }
)XXX");
  uint8_t t[3][3]{
      {0, 1, 2},
      {3, 0, 1},
      {2, 3, 0},
  };
  uint64_t luts[9][4]{
      {3, 0, 0, 0}, {0, 3, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 3}, {3, 0, 0, 0},
      {0, 3, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 3}, {3, 0, 0, 0},
  };
  uint64_t map[3][3]{
      {0, 1, 2},
      {3, 4, 5},
      {6, 7, 8},
  };
  uint8_t expected[3][3]{
      {3, 3, 3},
      {3, 3, 3},
      {3, 3, 3},
  };

  tensorArgTy<uint8_t> tArg(t);
  tensorArgTy<uint64_t> lutsArg(luts), mapArg(map);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&tArg, &lutsArg, &mapArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), 3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, apply_mapped_lookup_table_same_lut) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the lookup of 3x3 matrix of encrypted indices of width 2 of a 3x3 matrix of tables of size 4=2² of clear integers.
    func @main(%t: tensor<3x3x!FHE.eint<2>>, %luts: tensor<9x4xi64>, %map: tensor<3x3xindex>) -> tensor<3x3x!FHE.eint<2>> {
      %1 = "FHELinalg.apply_mapped_lookup_table"(%t, %luts, %map) :
        (tensor<3x3x!FHE.eint<2>>, tensor<9x4xi64>, tensor<3x3xindex>) -> tensor<3x3x!FHE.eint<2>>
      return %1: tensor<3x3x!FHE.eint<2>>
    }
)XXX");
  uint8_t t[3][3]{
      {0, 1, 2},
      {3, 0, 1},
      {2, 3, 0},
  };
  uint64_t luts[9][4]{
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 2, 3, 1},
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
  };
  uint64_t map[3][3]{
      {4, 4, 4},
      {4, 4, 4},
      {4, 4, 4},
  };
  uint8_t expected[3][3]{
      {1, 2, 3},
      {1, 1, 2},
      {3, 1, 1},
  };

  tensorArgTy<uint8_t> tArg(t);
  tensorArgTy<uint64_t> lutsArg(luts), mapArg(map);

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&tArg, &lutsArg, &mapArg});

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
// FHELinalg dot_eint_int ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(CompileAndRunTensorEncrypted, dot_eint_int_7) {
  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: tensor<4x!FHE.eint<7>>,
                   %arg1: tensor<4xi8>) -> !FHE.eint<7>
{
  %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<4x!FHE.eint<7>>, tensor<4xi8>) -> !FHE.eint<7>
  return %ret : !FHE.eint<7>
}
)XXX");
  static uint8_t arg0[] = {0, 1, 2, 3};
  static uint8_t arg1[] = {0, 1, 2, 3};

  llvm::Expected<uint64_t> res =
      lambda(arg0, ARRAY_SIZE(arg0), arg1, ARRAY_SIZE(arg1));

  ASSERT_EXPECTED_VALUE(res, 14);
}

///////////////////////////////////////////////////////////////////////////////
// FHELinalg neg_eint /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, neg_eint) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
    // Returns the negation of a 3x3 matrix of encrypted integers of width 2.
    //
    //        ([0,1,2])   [0,7,6]
    // negate ([3,4,5]) = [5,4,3]
    //        ([6,7,0])   [2,1,0]
    func @main(%t: tensor<3x3x!FHE.eint<2>>) -> tensor<3x3x!FHE.eint<2>> {
      %res = "FHELinalg.neg_eint"(%t) : (tensor<3x3x!FHE.eint<2>>) -> tensor<3x3x!FHE.eint<2>>
      return %res : tensor<3x3x!FHE.eint<2>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      tArg(llvm::ArrayRef<uint8_t>((const uint8_t *)t, 3 * 3), {3, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&tArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// FHELinalg matmul_eint_int ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, matmul_eint_int) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the matrix multiplication of a 3x2 matrix of encrypted integers and a 2x3 matrix of integers.
  //         [ 1, 2, 3]
  //         [ 2, 3, 4]
  //       *
  // [1,2]   [ 5, 8,11]
  // [3,4] = [11,18,25]
  // [5,6]   [17,28,39]
  func @main(%a: tensor<3x2x!FHE.eint<6>>, %b: tensor<2x3xi7>) -> tensor<3x3x!FHE.eint<6>> {
    %0 = "FHELinalg.matmul_eint_int"(%a, %b) : (tensor<3x2x!FHE.eint<6>>, tensor<2x3xi7>) -> tensor<3x3x!FHE.eint<6>>
    return %0 : tensor<3x3x!FHE.eint<6>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      aArg(llvm::ArrayRef<uint8_t>((const uint8_t *)A, 3 * 2), {3, 2});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      bArg(llvm::ArrayRef<uint8_t>((const uint8_t *)B, 2 * 3), {2, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&aArg, &bArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ((*res)[i * 3 + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// FHELinalg matmul_eint_int ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, matmul_int_eint) {

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
  // Returns the matrix multiplication of a 3x2 matrix of encrypted integers and a 2x3 matrix of integers.
  //         [ 1, 2, 3]
  //         [ 2, 3, 4]
  //       *
  // [1,2]   [ 5, 8,11]
  // [3,4] = [11,18,25]
  // [5,6]   [17,28,39]
  func @main(%a: tensor<3x2xi7>, %b: tensor<2x3x!FHE.eint<6>>) -> tensor<3x3x!FHE.eint<6>> {
    %0 = "FHELinalg.matmul_int_eint"(%a, %b) : (tensor<3x2xi7>, tensor<2x3x!FHE.eint<6>>) -> tensor<3x3x!FHE.eint<6>>
    return %0 : tensor<3x3x!FHE.eint<6>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      aArg(llvm::ArrayRef<uint8_t>((const uint8_t *)A, 3 * 2), {3, 2});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      bArg(llvm::ArrayRef<uint8_t>((const uint8_t *)B, 2 * 3), {2, 3});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&aArg, &bArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)3 * 3);

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

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%a: tensor<2x2x4x!FHE.eint<6>>) -> tensor<2x8x!FHE.eint<6>> {
  %0 = linalg.tensor_collapse_shape %a [[0],[1,2]] : tensor<2x2x4x!FHE.eint<6>> into tensor<2x8x!FHE.eint<6>>
  return %0 : tensor<2x8x!FHE.eint<6>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      aArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)A, 2 * 2 * 4), {2, 2, 4});

  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
          {&aArg});

  ASSERT_EXPECTED_SUCCESS(res);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<>> &resp =
      (*res)
          ->cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<>>>();

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

  mlir::concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%a: tensor<2x8x!FHE.eint<6>>) -> tensor<2x2x4x!FHE.eint<6>> {
  %0 = linalg.tensor_expand_shape %a [[0],[1,2]] : tensor<2x8x!FHE.eint<6>> into tensor<2x2x4x!FHE.eint<6>>
  return %0 : tensor<2x2x4x!FHE.eint<6>>
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

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      aArg(llvm::MutableArrayRef<uint8_t>((uint8_t *)A, 2 * 8), {2, 8});

  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>> res =
      lambda.operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
          {&aArg});

  ASSERT_EXPECTED_SUCCESS(res);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<>> &resp =
      (*res)
          ->cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<>>>();

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

///////////////////////////////////////////////////////////////////////////////
// FHELinalg sum /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_FHELinalg, sum_empty) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<0x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%x) : (tensor<0x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

)XXX");

  const uint8_t expected = 0;

  llvm::ArrayRef<uint8_t> xRef(nullptr, (size_t)0);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {0});

  llvm::Expected<uint64_t> result = lambda.operator()<uint64_t>({&xArg});
  ASSERT_EXPECTED_SUCCESS(result);

  ASSERT_EQ(*result, expected);
}

TEST(End2EndJit_FHELinalg, sum_1D_no_axes) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%x) : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

)XXX");

  const uint8_t x[4]{0, 1, 2, 3};
  const uint8_t expected = 6;

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {4});

  llvm::Expected<uint64_t> result = lambda.operator()<uint64_t>({&xArg});
  ASSERT_EXPECTED_SUCCESS(result);

  ASSERT_EQ(*result, expected);
}

TEST(End2EndJit_FHELinalg, sum_1D_axes_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%x) { axes = [0] } : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

)XXX");

  const uint8_t x[4]{0, 1, 2, 3};
  const uint8_t expected = 6;

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {4});

  llvm::Expected<uint64_t> result = lambda.operator()<uint64_t>({&xArg});
  ASSERT_EXPECTED_SUCCESS(result);

  ASSERT_EQ(*result, expected);
}

TEST(End2EndJit_FHELinalg, sum_2D_no_axes) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%x) : (tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

)XXX");

  const uint8_t x[3][4]{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 0, 1},
  };
  const uint8_t expected = 46;

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4});

  llvm::Expected<uint64_t> result = lambda.operator()<uint64_t>({&xArg});
  ASSERT_EXPECTED_SUCCESS(result);

  ASSERT_EQ(*result, expected);
}

TEST(End2EndJit_FHELinalg, sum_2D_axes_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0] } : (tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
  return %0 : tensor<4x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4]{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 0, 1},
  };
  const uint8_t expected[4]{12, 15, 8, 11};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 4);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, sum_2D_axes_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [1] } : (tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>
  return %0 : tensor<3x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4]{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 0, 1},
  };
  const uint8_t expected[3]{6, 22, 18};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 3);

  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, sum_2D_axes_0_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%x) { axes = [0, 1] } : (tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

)XXX");

  const uint8_t x[3][4]{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 0, 1},
  };
  const uint8_t expected = 46;

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4});

  llvm::Expected<uint64_t> result = lambda.operator()<uint64_t>({&xArg});
  ASSERT_EXPECTED_SUCCESS(result);

  ASSERT_EQ(*result, expected);
}

TEST(End2EndJit_FHELinalg, sum_3D_no_axes) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%x) : (tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected = 96;

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<uint64_t> result = lambda.operator()<uint64_t>({&xArg});
  ASSERT_EXPECTED_SUCCESS(result);

  ASSERT_EQ(*result, expected);
}

TEST(End2EndJit_FHELinalg, sum_3D_axes_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x2x!FHE.eint<7>>
  return %0 : tensor<4x2x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[4][2]{
      {14, 17},
      {10, 13},
      {6, 9},
      {12, 15},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 4);
  ASSERT_EQ(res.getDimensions().at(1), 2);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 8);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_EQ(res.getValue()[(i * 2) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_axes_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [1] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>>
  return %0 : tensor<3x2x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[3][2]{
      {12, 16},
      {14, 18},
      {16, 20},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EQ(res.getDimensions().at(1), 2);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 6);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_EQ(res.getValue()[(i * 2) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_axes_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [2] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>>
  return %0 : tensor<3x4x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[3][4]{
      {1, 5, 9, 13},
      {17, 1, 5, 9},
      {13, 17, 1, 5},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EQ(res.getDimensions().at(1), 4);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 12);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      EXPECT_EQ(res.getValue()[(i * 4) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_axes_0_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0, 1] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<2x!FHE.eint<7>>
  return %0 : tensor<2x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[2]{42, 54};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 2);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 2);

  for (size_t i = 0; i < 2; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_axes_1_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [1, 2] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>
  return %0 : tensor<3x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[3]{28, 32, 36};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 3);

  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_axes_0_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0, 2] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
  return %0 : tensor<4x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[4]{31, 23, 15, 27};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 4);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 4);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_axes_0_1_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%x) { axes = [0, 1, 2] } : (tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected = 96;

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<uint64_t> result = lambda.operator()<uint64_t>({&xArg});
  ASSERT_EXPECTED_SUCCESS(result);

  ASSERT_EQ(*result, expected);
}

TEST(End2EndJit_FHELinalg, sum_keep_dims_empty) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<0x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { keep_dims = true } : (tensor<0x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
  return %0 : tensor<1x!FHE.eint<7>>
}

)XXX");

  const uint8_t expected[1] = {0};

  llvm::ArrayRef<uint8_t> xRef(nullptr, (size_t)0);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {0});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 1);

  for (size_t i = 0; i < 1; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, sum_1D_keep_dims_no_axes) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { keep_dims = true } : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
  return %0 : tensor<1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[4]{0, 1, 2, 3};
  const uint8_t expected[1] = {6};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 1);

  for (size_t i = 0; i < 1; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, sum_1D_keep_dims_axes_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0], keep_dims = true } : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
  return %0 : tensor<1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[4]{0, 1, 2, 3};
  const uint8_t expected[1] = {6};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 1);

  for (size_t i = 0; i < 1; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, sum_2D_keep_dims_no_axes) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4]{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 0, 1},
  };
  const uint8_t expected[1][1] = {{46}};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EQ(res.getDimensions().at(1), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 1);

  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 1; j++) {
      EXPECT_EQ(res.getValue()[(i * 1) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_2D_keep_dims_axes_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>>
  return %0 : tensor<1x4x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4]{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 0, 1},
  };
  const uint8_t expected[1][4]{
      {12, 15, 8, 11},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EQ(res.getDimensions().at(1), 4);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 4);

  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 4; j++) {
      EXPECT_EQ(res.getValue()[(i * 4) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_2D_keep_dims_axes_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [1], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>>
  return %0 : tensor<3x1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4]{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 0, 1},
  };
  const uint8_t expected[3][1]{
      {6},
      {22},
      {18},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EQ(res.getDimensions().at(1), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 1; j++) {
      EXPECT_EQ(res.getValue()[(i * 1) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_2D_keep_dims_axes_0_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0, 1], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4]{
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 0, 1},
  };
  const uint8_t expected[1][1] = {{46}};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EQ(res.getDimensions().at(1), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 1);

  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 1; j++) {
      EXPECT_EQ(res.getValue()[(i * 1) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_keep_dims_no_axes) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[1][1][1] = {{{96}}};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EQ(res.getDimensions().at(1), 1);
  ASSERT_EQ(res.getDimensions().at(2), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 1);

  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 1; j++) {
      for (size_t k = 0; k < 1; k++) {
        EXPECT_EQ(res.getValue()[(i * 1 * 1) + (j * 1) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_keep_dims_axes_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x2x!FHE.eint<7>>
  return %0 : tensor<1x4x2x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[1][4][2]{{
      {14, 17},
      {10, 13},
      {6, 9},
      {12, 15},
  }};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EQ(res.getDimensions().at(1), 4);
  ASSERT_EQ(res.getDimensions().at(2), 2);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 8);

  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 2; k++) {
        EXPECT_EQ(res.getValue()[(i * 4 * 2) + (j * 2) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_keep_dims_axes_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [1], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>>
  return %0 : tensor<3x1x2x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[3][1][2]{
      {{12, 16}},
      {{14, 18}},
      {{16, 20}},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EQ(res.getDimensions().at(1), 1);
  ASSERT_EQ(res.getDimensions().at(2), 2);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 6);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 1; j++) {
      for (size_t k = 0; k < 2; k++) {
        EXPECT_EQ(res.getValue()[(i * 1 * 2) + (j * 2) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_keep_dims_axes_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x4x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [2], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x4x1x!FHE.eint<7>>
  return %0 : tensor<3x4x1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[3][4][1]{
      {{1}, {5}, {9}, {13}},
      {{17}, {1}, {5}, {9}},
      {{13}, {17}, {1}, {5}},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EQ(res.getDimensions().at(1), 4);
  ASSERT_EQ(res.getDimensions().at(2), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 12);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 1; k++) {
        EXPECT_EQ(res.getValue()[(i * 4 * 1) + (j * 1) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_keep_dims_axes_0_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0, 1], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x2x!FHE.eint<7>>
  return %0 : tensor<1x1x2x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[1][1][2]{{{42, 54}}};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EQ(res.getDimensions().at(1), 1);
  ASSERT_EQ(res.getDimensions().at(2), 2);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 2);

  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 1; j++) {
      for (size_t k = 0; k < 2; k++) {
        EXPECT_EQ(res.getValue()[(i * 1 * 2) + (j * 2) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_keep_dims_axes_1_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [1, 2], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x1x!FHE.eint<7>>
  return %0 : tensor<3x1x1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[3][1][1]{{{28}}, {{32}}, {{36}}};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EQ(res.getDimensions().at(1), 1);
  ASSERT_EQ(res.getDimensions().at(2), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 3);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 1; j++) {
      for (size_t k = 0; k < 1; k++) {
        EXPECT_EQ(res.getValue()[(i * 1 * 1) + (j * 1) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_keep_dims_axes_0_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0, 2], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>>
  return %0 : tensor<1x4x1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[1][4][1]{{{31}, {23}, {15}, {27}}};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EQ(res.getDimensions().at(1), 4);
  ASSERT_EQ(res.getDimensions().at(2), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 4);

  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 1; k++) {
        EXPECT_EQ(res.getValue()[(i * 4 * 1) + (j * 1) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, sum_3D_keep_dims_axes_0_1_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(

func @main(%x: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%x) { axes = [0, 1, 2], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x1x!FHE.eint<7>>
}

)XXX");

  const uint8_t x[3][4][2]{
      {
          {0, 1},
          {2, 3},
          {4, 5},
          {6, 7},
      },
      {
          {8, 9},
          {0, 1},
          {2, 3},
          {4, 5},
      },
      {
          {6, 7},
          {8, 9},
          {0, 1},
          {2, 3},
      },
  };
  const uint8_t expected[1][1][1] = {{{96}}};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 4 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 4, 2});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>({&xArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 1);
  ASSERT_EQ(res.getDimensions().at(1), 1);
  ASSERT_EQ(res.getDimensions().at(2), 1);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 1);

  for (size_t i = 0; i < 1; i++) {
    for (size_t j = 0; j < 1; j++) {
      for (size_t k = 0; k < 1; k++) {
        EXPECT_EQ(res.getValue()[(i * 1 * 1) + (j * 1) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, concat_1D_axis_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%x: tensor<3x!FHE.eint<7>>, %y: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<3x!FHE.eint<7>>, tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}
)XXX");

  const uint8_t x[3]{0, 1, 2};
  const uint8_t y[4]{3, 4, 5, 6};

  const uint8_t expected[7]{0, 1, 2, 3, 4, 5, 6};

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3});

  llvm::ArrayRef<uint8_t> yRef((const uint8_t *)y, 4);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      yArg(yRef, {4});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>(
          {&xArg, &yArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)1);
  ASSERT_EQ(res.getDimensions().at(0), 7);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 7);

  for (size_t i = 0; i < 7; i++) {
    EXPECT_EQ(res.getValue()[i], expected[i]) << ", at pos(" << i << ")";
  }
}

TEST(End2EndJit_FHELinalg, concat_2D_axis_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%x: tensor<2x3x!FHE.eint<7>>, %y: tensor<3x3x!FHE.eint<7>>) -> tensor<5x3x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<2x3x!FHE.eint<7>>, tensor<3x3x!FHE.eint<7>>) -> tensor<5x3x!FHE.eint<7>>
  return %0 : tensor<5x3x!FHE.eint<7>>
}
)XXX");

  const uint8_t x[2][3]{
      {0, 1, 2},
      {3, 4, 5},
  };
  const uint8_t y[3][3]{
      {6, 7, 8},
      {9, 0, 1},
      {2, 3, 4},
  };

  const uint8_t expected[5][3]{
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 0, 1}, {2, 3, 4},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 2 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {2, 3});

  llvm::ArrayRef<uint8_t> yRef((const uint8_t *)y, 3 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      yArg(yRef, {3, 3});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>(
          {&xArg, &yArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 5);
  ASSERT_EQ(res.getDimensions().at(1), 3);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 15);

  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(res.getValue()[(i * 3) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, concat_2D_axis_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%x: tensor<3x2x!FHE.eint<7>>, %y: tensor<3x3x!FHE.eint<7>>) -> tensor<3x5x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<3x2x!FHE.eint<7>>, tensor<3x3x!FHE.eint<7>>) -> tensor<3x5x!FHE.eint<7>>
  return %0 : tensor<3x5x!FHE.eint<7>>
}
)XXX");

  const uint8_t x[3][2]{
      {0, 1},
      {2, 3},
      {4, 5},
  };
  const uint8_t y[3][3]{
      {6, 7, 8},
      {9, 0, 1},
      {2, 3, 4},
  };

  const uint8_t expected[3][5]{
      {0, 1, 6, 7, 8},
      {2, 3, 9, 0, 1},
      {4, 5, 2, 3, 4},
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 3 * 2);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {3, 2});

  llvm::ArrayRef<uint8_t> yRef((const uint8_t *)y, 3 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      yArg(yRef, {3, 3});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>(
          {&xArg, &yArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)2);
  ASSERT_EQ(res.getDimensions().at(0), 3);
  ASSERT_EQ(res.getDimensions().at(1), 5);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 15);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 5; j++) {
      EXPECT_EQ(res.getValue()[(i * 5) + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

TEST(End2EndJit_FHELinalg, concat_3D_axis_0) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%x: tensor<2x4x3x!FHE.eint<7>>, %y: tensor<2x4x3x!FHE.eint<7>>) -> tensor<4x4x3x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<2x4x3x!FHE.eint<7>>, tensor<2x4x3x!FHE.eint<7>>) -> tensor<4x4x3x!FHE.eint<7>>
  return %0 : tensor<4x4x3x!FHE.eint<7>>
}
)XXX");

  const uint8_t x[2][4][3]{
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
  };
  const uint8_t y[2][4][3]{
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
  };

  const uint8_t expected[4][4][3]{
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 2 * 4 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {2, 4, 3});

  llvm::ArrayRef<uint8_t> yRef((const uint8_t *)y, 2 * 4 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      yArg(yRef, {2, 4, 3});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>(
          {&xArg, &yArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 4);
  ASSERT_EQ(res.getDimensions().at(1), 4);
  ASSERT_EQ(res.getDimensions().at(2), 3);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 48);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 3; k++) {
        EXPECT_EQ(res.getValue()[(i * 4 * 3) + (j * 3) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, concat_3D_axis_1) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%x: tensor<2x4x3x!FHE.eint<7>>, %y: tensor<2x4x3x!FHE.eint<7>>) -> tensor<2x8x3x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<2x4x3x!FHE.eint<7>>, tensor<2x4x3x!FHE.eint<7>>) -> tensor<2x8x3x!FHE.eint<7>>
  return %0 : tensor<2x8x3x!FHE.eint<7>>
}
)XXX");

  const uint8_t x[2][4][3]{
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
  };
  const uint8_t y[2][4][3]{
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
  };

  const uint8_t expected[2][8][3]{
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 2 * 4 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {2, 4, 3});

  llvm::ArrayRef<uint8_t> yRef((const uint8_t *)y, 2 * 4 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      yArg(yRef, {2, 4, 3});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>(
          {&xArg, &yArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 2);
  ASSERT_EQ(res.getDimensions().at(1), 8);
  ASSERT_EQ(res.getDimensions().at(2), 3);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 48);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 8; j++) {
      for (size_t k = 0; k < 3; k++) {
        EXPECT_EQ(res.getValue()[(i * 8 * 3) + (j * 3) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

TEST(End2EndJit_FHELinalg, concat_3D_axis_2) {
  namespace concretelang = mlir::concretelang;

  concretelang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%x: tensor<2x4x3x!FHE.eint<7>>, %y: tensor<2x4x3x!FHE.eint<7>>) -> tensor<2x4x6x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 2 } : (tensor<2x4x3x!FHE.eint<7>>, tensor<2x4x3x!FHE.eint<7>>) -> tensor<2x4x6x!FHE.eint<7>>
  return %0 : tensor<2x4x6x!FHE.eint<7>>
}
)XXX");

  const uint8_t x[2][4][3]{
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
  };
  const uint8_t y[2][4][3]{
      {
          {0, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
      },
      {
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
      },
  };

  const uint8_t expected[2][4][6]{
      {
          {0, 1, 2, 0, 1, 2},
          {3, 4, 5, 3, 4, 5},
          {6, 7, 8, 6, 7, 8},
          {9, 0, 1, 9, 0, 1},
      },
      {
          {2, 3, 4, 2, 3, 4},
          {5, 6, 7, 5, 6, 7},
          {8, 9, 0, 8, 9, 0},
          {1, 2, 3, 1, 2, 3},
      },
  };

  llvm::ArrayRef<uint8_t> xRef((const uint8_t *)x, 2 * 4 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      xArg(xRef, {2, 4, 3});

  llvm::ArrayRef<uint8_t> yRef((const uint8_t *)y, 2 * 4 * 3);
  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<uint8_t>>
      yArg(yRef, {2, 4, 3});

  llvm::Expected<std::unique_ptr<concretelang::LambdaArgument>> call =
      lambda.operator()<std::unique_ptr<concretelang::LambdaArgument>>(
          {&xArg, &yArg});
  ASSERT_EXPECTED_SUCCESS(call);

  concretelang::TensorLambdaArgument<concretelang::IntLambdaArgument<>> &res =
      (*call)
          ->cast<concretelang::TensorLambdaArgument<
              concretelang::IntLambdaArgument<>>>();

  ASSERT_EQ(res.getDimensions().size(), (size_t)3);
  ASSERT_EQ(res.getDimensions().at(0), 2);
  ASSERT_EQ(res.getDimensions().at(1), 4);
  ASSERT_EQ(res.getDimensions().at(2), 6);
  ASSERT_EXPECTED_VALUE(res.getNumElements(), 48);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 6; k++) {
        EXPECT_EQ(res.getValue()[(i * 4 * 6) + (j * 6) + k], expected[i][j][k])
            << ", at pos(" << i << "," << j << "," << k << ")";
      }
    }
  }
}

class TiledMatMulParametric
    : public ::testing::TestWithParam<std::vector<int64_t>> {};

TEST_P(TiledMatMulParametric, tiled_matmul_eint_int) {
  std::vector<int64_t> tiling = GetParam();
  std::ostringstream mlirProgram;

  mlirProgram
      << "func @main(%a: tensor<8x4x!FHE.eint<6>>, %b: tensor<4x2xi7>) ->\n"
      << "  tensor<8x2x!FHE.eint<6>> {\n"
      << "    %0 = \"FHELinalg.matmul_eint_int\"(%a, %b) { \"tile-sizes\" = "
      << "[" << tiling[0] << ", " << tiling[1] << ", " << tiling[2] << "]"
      << "} :\n"
      << "           (tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) ->\n"
      << "           tensor<8x2x!FHE.eint<6>>\n"
      << "    return %0 : tensor<8x2x!FHE.eint<6>>\n"
      << "  }";

  mlir::concretelang::JitCompilerEngine::Lambda lambda =
      checkedJit(mlirProgram.str());

  const size_t rowsA = 8;
  const size_t colsA = 4;
  const uint8_t A[rowsA][colsA] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 0, 1, 2},
                                   {3, 4, 5, 6}, {7, 8, 9, 0}, {1, 2, 3, 4},
                                   {5, 6, 7, 8}, {9, 0, 1, 2}};

  const size_t rowsB = 4;
  const size_t colsB = 2;
  const uint8_t B[rowsB][colsB]{{1, 2}, {3, 4}, {3, 1}, {0, 2}};

  const size_t rowsC = rowsA;
  const size_t colsC = colsB;
  const uint8_t expected[rowsC][colsC]{
      {16, 21}, {44, 57}, {12, 23}, {30, 39},
      {58, 55}, {16, 21}, {44, 57}, {12, 23},
  };

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      aArg(llvm::ArrayRef<uint8_t>((const uint8_t *)A, rowsA * colsA),
           {rowsA, colsA});
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      bArg(llvm::ArrayRef<uint8_t>((const uint8_t *)B, rowsB * colsB),
           {rowsB, colsB});

  llvm::Expected<std::vector<uint64_t>> res =
      lambda.operator()<std::vector<uint64_t>>({&aArg, &bArg});

  ASSERT_EXPECTED_SUCCESS(res);

  ASSERT_EQ(res->size(), (uint64_t)(rowsC * colsC));

  for (size_t i = 0; i < rowsC; i++) {
    for (size_t j = 0; j < colsC; j++) {
      EXPECT_EQ((*res)[i * colsC + j], expected[i][j])
          << ", at pos(" << i << "," << j << ")";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DISABLED_TiledMatMul, TiledMatMulParametric,
                         ::testing::Values(
                             // Element-sized tiles
                             std::vector<int64_t>{1, 1, 1},

                             // Mixed tiles
                             std::vector<int64_t>{2, 2, 2},
                             std::vector<int64_t>{4, 4, 2},
                             std::vector<int64_t>{2, 4, 2},

                             // Single, big tile
                             std::vector<int64_t>{8, 4, 2}));
