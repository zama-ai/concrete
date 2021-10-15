#include "end_to_end_jit_test.h"

///////////////////////////////////////////////////////////////////////////////
// 1D tensor //////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(End2EndJit_StdTensor_1D, identity) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi64>) -> tensor<10xi64> {
  return %t : tensor<10xi64>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  const size_t size = 10;
  uint64_t arg[size]{0xFFFFFFFFFFFFFFFF,
                     0,
                     8978,
                     2587490,
                     90,
                     197864,
                     698735,
                     72132,
                     87474,
                     42};
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, arg, size));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t result[size];
  ASSERT_LLVM_ERROR(argument->getResult(0, result, size));
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(arg[i], result[i]) << "result differ at index " << i;
  }
}

TEST(End2EndJit_StdTensor_1D, extract_64) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi64>, %i: index) -> i64{
  %c = tensor.extract %t[%i] : tensor<10xi64>
  return %c : i64
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  const size_t size = 10;
  uint64_t t_arg[size]{0xFFFFFFFFFFFFFFFF,
                       0,
                       8978,
                       2587490,
                       90,
                       197864,
                       698735,
                       72132,
                       87474,
                       42};
  for (size_t i = 0; i < size; i++) {
    auto maybeArgument = engine.buildArgument();
    ASSERT_LLVM_ERROR(maybeArgument.takeError());
    auto argument = std::move(maybeArgument.get());
    // Set the %t argument
    ASSERT_LLVM_ERROR(argument->setArg(0, t_arg, size));
    // Set the %i argument
    ASSERT_LLVM_ERROR(argument->setArg(1, i));
    // Invoke the function
    ASSERT_LLVM_ERROR(engine.invoke(*argument));
    // Get and assert the result
    uint64_t res = 0;
    ASSERT_LLVM_ERROR(argument->getResult(0, res));
    ASSERT_EQ(res, t_arg[i]);
  }
}

TEST(End2EndJit_StdTensor_1D, extract_32) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi32>, %i: index) -> i32{
  %c = tensor.extract %t[%i] : tensor<10xi32>
  return %c : i32
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  const size_t size = 10;
  uint32_t t_arg[size]{0xFFFFFFFF, 0,      8978,  2587490, 90,
                       197864,     698735, 72132, 87474,   42};
  for (size_t i = 0; i < size; i++) {
    auto maybeArgument = engine.buildArgument();
    ASSERT_LLVM_ERROR(maybeArgument.takeError());
    auto argument = std::move(maybeArgument.get());
    // Set the %t argument
    ASSERT_LLVM_ERROR(argument->setArg(0, t_arg, size));
    // Set the %i argument
    ASSERT_LLVM_ERROR(argument->setArg(1, i));
    // Invoke the function
    ASSERT_LLVM_ERROR(engine.invoke(*argument));
    // Get and assert the result
    uint64_t res = 0;
    ASSERT_LLVM_ERROR(argument->getResult(0, res));
    ASSERT_EQ(res, t_arg[i]);
  }
}

TEST(End2EndJit_StdTensor_1D, extract_16) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi16>, %i: index) -> i16{
  %c = tensor.extract %t[%i] : tensor<10xi16>
  return %c : i16
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  const size_t size = 10;
  uint16_t t_arg[size]{0xFFFF, 0,     59589, 47826, 16227,
                       63269,  36435, 52380, 7401,  13313};
  for (size_t i = 0; i < size; i++) {
    auto maybeArgument = engine.buildArgument();
    ASSERT_LLVM_ERROR(maybeArgument.takeError());
    auto argument = std::move(maybeArgument.get());
    // Set the %t argument
    ASSERT_LLVM_ERROR(argument->setArg(0, t_arg, size));
    // Set the %i argument
    ASSERT_LLVM_ERROR(argument->setArg(1, i));
    // Invoke the function
    ASSERT_LLVM_ERROR(engine.invoke(*argument));
    // Get and assert the result
    uint64_t res = 0;
    ASSERT_LLVM_ERROR(argument->getResult(0, res));
    ASSERT_EQ(res, t_arg[i]);
  }
}

TEST(End2EndJit_StdTensor_1D, extract_8) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi8>, %i: index) -> i8{
  %c = tensor.extract %t[%i] : tensor<10xi8>
  return %c : i8
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  const size_t size = 10;
  uint8_t t_arg[size]{0xFF, 0, 120, 225, 14, 177, 131, 84, 174, 93};
  for (size_t i = 0; i < size; i++) {
    auto maybeArgument = engine.buildArgument();
    ASSERT_LLVM_ERROR(maybeArgument.takeError());
    auto argument = std::move(maybeArgument.get());
    // Set the %t argument
    ASSERT_LLVM_ERROR(argument->setArg(0, t_arg, size));
    // Set the %i argument
    ASSERT_LLVM_ERROR(argument->setArg(1, i));
    // Invoke the function
    ASSERT_LLVM_ERROR(engine.invoke(*argument));
    // Get and assert the result
    uint64_t res = 0;
    ASSERT_LLVM_ERROR(argument->getResult(0, res));
    ASSERT_EQ(res, t_arg[i]);
  }
}

TEST(End2EndJit_StdTensor_1D, extract_5) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi5>, %i: index) -> i5{
  %c = tensor.extract %t[%i] : tensor<10xi5>
  return %c : i5
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  const size_t size = 10;
  uint8_t t_arg[size]{32, 0, 10, 25, 14, 25, 18, 28, 14, 7};
  for (size_t i = 0; i < size; i++) {
    auto maybeArgument = engine.buildArgument();
    ASSERT_LLVM_ERROR(maybeArgument.takeError());
    auto argument = std::move(maybeArgument.get());
    // Set the %t argument
    ASSERT_LLVM_ERROR(argument->setArg(0, t_arg, size));
    // Set the %i argument
    ASSERT_LLVM_ERROR(argument->setArg(1, i));
    // Invoke the function
    ASSERT_LLVM_ERROR(engine.invoke(*argument));
    // Get and assert the result
    uint64_t res = 0;
    ASSERT_LLVM_ERROR(argument->getResult(0, res));
    ASSERT_EQ(res, t_arg[i]);
  }
}

TEST(End2EndJit_StdTensor_1D, extract_1) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi1>, %i: index) -> i1{
  %c = tensor.extract %t[%i] : tensor<10xi1>
  return %c : i1
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  const size_t size = 10;
  uint8_t t_arg[size]{0, 0, 1, 0, 1, 1, 0, 1, 1, 0};
  for (size_t i = 0; i < size; i++) {
    auto maybeArgument = engine.buildArgument();
    ASSERT_LLVM_ERROR(maybeArgument.takeError());
    auto argument = std::move(maybeArgument.get());
    // Set the %t argument
    ASSERT_LLVM_ERROR(argument->setArg(0, t_arg, size));
    // Set the %i argument
    ASSERT_LLVM_ERROR(argument->setArg(1, i));
    // Invoke the function
    ASSERT_LLVM_ERROR(engine.invoke(*argument));
    // Get and assert the result
    uint64_t res = 0;
    ASSERT_LLVM_ERROR(argument->getResult(0, res));
    ASSERT_EQ(res, t_arg[i]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// 2D tensor //////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

const size_t numDim = 2;
const size_t dim0 = 2;
const size_t dim1 = 10;
const size_t dims[numDim]{dim0, dim1};
const uint64_t tensor2D[dim0][dim1]{
    {0xFFFFFFFFFFFFFFFF, 0, 8978, 2587490, 90, 197864, 698735, 72132, 87474,
     42},
    {986, 1873, 298493, 34939, 443, 59874, 43, 743, 8409, 9433},
};

TEST(End2EndJit_StdTensor_2D, identity) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<2x10xi64>) -> tensor<2x10xi64> {
  return %t : tensor<2x10xi64>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));

  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint64_t *)tensor2D, numDim, dims));
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

TEST(End2EndJit_StdTensor_2D, extract) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<2x10xi64>, %i: index, %j: index) -> i64 {
  %c = tensor.extract %t[%i, %j] : tensor<2x10xi64>
  return %c : i64
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint64_t *)tensor2D, numDim, dims));
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

TEST(End2EndJit_StdTensor_2D, extract_slice) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<2x10xi64>) -> tensor<1x5xi64> {
  %r = tensor.extract_slice %t[1, 5][1, 5][1, 1] : tensor<2x10xi64> to tensor<1x5xi64>
  return %r : tensor<1x5xi64>
}
)XXX";

  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint64_t *)tensor2D, numDim, dims));
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

TEST(End2EndJit_StdTensor_2D, extract_slice_stride) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<2x10xi64>) -> tensor<1x5xi64> {
  %r = tensor.extract_slice %t[1, 0][1, 5][1, 2] : tensor<2x10xi64> to tensor<1x5xi64>
  return %r : tensor<1x5xi64>
}
)XXX";

  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint64_t *)tensor2D, numDim, dims));
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

TEST(End2EndJit_StdTensor_2D, insert_slice) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t0: tensor<2x10xi64>, %t1: tensor<2x2xi64>) -> tensor<2x10xi64> {
  %r = tensor.insert_slice %t1 into %t0[0, 5][2, 2][1, 1] : tensor<2x2xi64> into tensor<2x10xi64>
  return %r : tensor<2x10xi64>
}
)XXX";

  ASSERT_LLVM_ERROR(engine.compile(mlirStr, defaultV0Constraints()));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t0 argument
  ASSERT_LLVM_ERROR(argument->setArg(0, (uint64_t *)tensor2D, numDim, dims));
  // Set the %t1 argument
  uint64_t t1_dim[2] = {2, 2};
  uint64_t t1[2][2]{{6, 9}, {4, 0}};
  ASSERT_LLVM_ERROR(argument->setArg(1, (uint64_t *)t1, 2, t1_dim));
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