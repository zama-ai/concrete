#include <gtest/gtest.h>

#include "zamalang/Support/CompilerEngine.h"

#define ASSERT_LLVM_ERROR(err)                                                 \
  if (err) {                                                                   \
    llvm::errs() << "error: " << std::move(err) << "\n";                       \
    ASSERT_TRUE(false);                                                        \
  }

TEST(CompileAndRunHLFHE, add_eint) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%arg0: !HLFHE.eint<6>, %arg1: !HLFHE.eint<6>) -> !HLFHE.eint<6> {
  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<6>, !HLFHE.eint<6>) -> (!HLFHE.eint<6>)
  return %1: !HLFHE.eint<6>
}
)XXX";
  ASSERT_FALSE(engine.compile(mlirStr));
  auto maybeResult = engine.run({1, 2});
  ASSERT_TRUE((bool)maybeResult);
  uint64_t result = maybeResult.get();
  ASSERT_EQ(result, 3);
}

TEST(CompileAndRunTensorStd, extract_64) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi64>, %i: index) -> i64{
  %c = tensor.extract %t[%i] : tensor<10xi64>
  return %c : i64
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
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

TEST(CompileAndRunTensorStd, extract_32) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi32>, %i: index) -> i32{
  %c = tensor.extract %t[%i] : tensor<10xi32>
  return %c : i32
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
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

TEST(CompileAndRunTensorStd, extract_16) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi16>, %i: index) -> i16{
  %c = tensor.extract %t[%i] : tensor<10xi16>
  return %c : i16
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
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

TEST(CompileAndRunTensorStd, extract_8) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi8>, %i: index) -> i8{
  %c = tensor.extract %t[%i] : tensor<10xi8>
  return %c : i8
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
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

TEST(CompileAndRunTensorStd, extract_5) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi5>, %i: index) -> i5{
  %c = tensor.extract %t[%i] : tensor<10xi5>
  return %c : i5
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
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

TEST(CompileAndRunTensorStd, extract_1) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10xi1>, %i: index) -> i1{
  %c = tensor.extract %t[%i] : tensor<10xi1>
  return %c : i1
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
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

TEST(CompileAndRunTensorEncrypted, extract_5) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10x!HLFHE.eint<5>>, %i: index) -> !HLFHE.eint<5>{
  %c = tensor.extract %t[%i] : tensor<10x!HLFHE.eint<5>>
  return %c : !HLFHE.eint<5>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
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

TEST(CompileAndRunTensorEncrypted, extract_twice_and_add_5) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10x!HLFHE.eint<5>>, %i: index, %j: index) -> !HLFHE.eint<5>{
  %ti = tensor.extract %t[%i] : tensor<10x!HLFHE.eint<5>>
  %tj = tensor.extract %t[%j] : tensor<10x!HLFHE.eint<5>>
  %c = "HLFHE.add_eint"(%ti, %tj) : (!HLFHE.eint<5>, !HLFHE.eint<5>) -> !HLFHE.eint<5>
  return %c : !HLFHE.eint<5>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  const size_t size = 10;
  uint8_t t_arg[size]{32, 0, 10, 25, 14, 25, 18, 28, 14, 7};
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      auto maybeArgument = engine.buildArgument();
      ASSERT_LLVM_ERROR(maybeArgument.takeError());
      auto argument = std::move(maybeArgument.get());
      // Set the %t argument
      ASSERT_LLVM_ERROR(argument->setArg(0, t_arg, size));
      // Set the %i argument
      ASSERT_LLVM_ERROR(argument->setArg(1, i));
      // Set the %j argument
      ASSERT_LLVM_ERROR(argument->setArg(2, j));
      // Invoke the function
      ASSERT_LLVM_ERROR(engine.invoke(*argument));
      // Get and assert the result
      uint64_t res = 0;
      ASSERT_LLVM_ERROR(argument->getResult(0, res));
      ASSERT_EQ(res, t_arg[i] + t_arg[j]);
    }
  }
}

TEST(CompileAndRunTensorEncrypted, dim_5) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%t: tensor<10x!HLFHE.eint<5>>) -> index{
  %c0 = constant 0 : index
  %c = tensor.dim %t, %c0 : tensor<10x!HLFHE.eint<5>>
  return %c : index
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  const size_t size = 10;
  uint8_t t_arg[size]{32, 0, 10, 25, 14, 25, 18, 28, 14, 7};
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, t_arg, size));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t res = 0;
  ASSERT_LLVM_ERROR(argument->getResult(0, res));
  ASSERT_EQ(res, size);
}

TEST(CompileAndRunTensorEncrypted, from_elements_5) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%0: !HLFHE.eint<5>) -> tensor<1x!HLFHE.eint<5>> {
  %t = tensor.from_elements %0 : tensor<1x!HLFHE.eint<5>>
  return %t: tensor<1x!HLFHE.eint<5>>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the %t argument
  ASSERT_LLVM_ERROR(argument->setArg(0, 10));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  size_t size_res = 1;
  uint64_t t_res[size_res];
  ASSERT_LLVM_ERROR(argument->getResult(0, t_res, size_res));
  ASSERT_EQ(t_res[0], 10);
}

TEST(CompileAndRunTensorEncrypted, in_out_tensor_with_op_5) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%in: tensor<2x!HLFHE.eint<5>>) -> tensor<3x!HLFHE.eint<5>> {
  %c_0 = constant 0 : index
  %c_1 = constant 1 : index
  %a = tensor.extract %in[%c_0] : tensor<2x!HLFHE.eint<5>>
  %b = tensor.extract %in[%c_1] : tensor<2x!HLFHE.eint<5>>
  %aplusa = "HLFHE.add_eint"(%a, %a): (!HLFHE.eint<5>, !HLFHE.eint<5>) -> (!HLFHE.eint<5>)
  %aplusb = "HLFHE.add_eint"(%a, %b): (!HLFHE.eint<5>, !HLFHE.eint<5>) -> (!HLFHE.eint<5>)
  %bplusb = "HLFHE.add_eint"(%b, %b): (!HLFHE.eint<5>, !HLFHE.eint<5>) -> (!HLFHE.eint<5>)
  %out = tensor.from_elements %aplusa, %aplusb, %bplusb : tensor<3x!HLFHE.eint<5>>
  return %out: tensor<3x!HLFHE.eint<5>>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set the argument
  const size_t in_size = 2;
  uint8_t in[in_size] = {2, 16};
  ASSERT_LLVM_ERROR(argument->setArg(0, in, in_size));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  const size_t size_res = 3;
  uint64_t t_res[size_res];
  ASSERT_LLVM_ERROR(argument->getResult(0, t_res, size_res));
  ASSERT_EQ(t_res[0], in[0] + in[0]);
  ASSERT_EQ(t_res[1], in[0] + in[1]);
  ASSERT_EQ(t_res[2], in[1] + in[1]);
}

TEST(CompileAndRunTensorEncrypted, linalg_generic) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
func @main(%arg0: tensor<2x!HLFHE.eint<6>>, %arg1: tensor<2xi7>, %acc: !HLFHE.eint<6>) -> !HLFHE.eint<6> {
  %tacc = tensor.from_elements %acc : tensor<1x!HLFHE.eint<6>>
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!HLFHE.eint<6>>, tensor<2xi7>) outs(%tacc : tensor<1x!HLFHE.eint<6>>) {
  ^bb0(%arg2: !HLFHE.eint<6>, %arg3: i7, %arg4: !HLFHE.eint<6>):  // no predecessors
    %4 = "HLFHE.mul_eint_int"(%arg2, %arg3) : (!HLFHE.eint<6>, i7) -> !HLFHE.eint<6>
    %5 = "HLFHE.add_eint"(%4, %arg4) : (!HLFHE.eint<6>, !HLFHE.eint<6>) -> !HLFHE.eint<6>
    linalg.yield %5 : !HLFHE.eint<6>
  } -> tensor<1x!HLFHE.eint<6>>
  %c0 = constant 0 : index
  %ret = tensor.extract %2[%c0] : tensor<1x!HLFHE.eint<6>>
  return %ret : !HLFHE.eint<6>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set arg0, arg1, acc
  const size_t in_size = 2;
  uint8_t arg0[in_size] = {2, 8};
  ASSERT_LLVM_ERROR(argument->setArg(0, arg0, in_size));
  uint8_t arg1[in_size] = {6, 8};
  ASSERT_LLVM_ERROR(argument->setArg(1, arg1, in_size));
  ASSERT_LLVM_ERROR(argument->setArg(2, 0));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t res;
  ASSERT_LLVM_ERROR(argument->getResult(0, res));
  ASSERT_EQ(res, 76);
}

TEST(CompileAndRunTensorEncrypted, dot_eint_int_7) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%arg0: tensor<4x!HLFHE.eint<6>>,
                   %arg1: tensor<4xi7>) -> !HLFHE.eint<6>
{
  %ret = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<4x!HLFHE.eint<6>>, tensor<4xi7>) -> !HLFHE.eint<6>
  return %ret : !HLFHE.eint<6>
}
)XXX";
  ASSERT_LLVM_ERROR(engine.compile(mlirStr));
  auto maybeArgument = engine.buildArgument();
  ASSERT_LLVM_ERROR(maybeArgument.takeError());
  auto argument = std::move(maybeArgument.get());
  // Set arg0, arg1, acc
  const size_t in_size = 4;
  uint8_t arg0[in_size] = {0, 1, 2, 3};
  ASSERT_LLVM_ERROR(argument->setArg(0, arg0, in_size));
  uint8_t arg1[in_size] = {0, 1, 2, 3};
  ASSERT_LLVM_ERROR(argument->setArg(1, arg1, in_size));
  // Invoke the function
  ASSERT_LLVM_ERROR(engine.invoke(*argument));
  // Get and assert the result
  uint64_t res;
  ASSERT_LLVM_ERROR(argument->getResult(0, res));
  ASSERT_EQ(res, 14);
}

TEST(CompileAndRunTLU, identity_func) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%arg0: !HLFHE.eint<6>) -> !HLFHE.eint<6> {
    %tlu = std.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
    %1 = "HLFHE.apply_lookup_table"(%arg0, %tlu): (!HLFHE.eint<6>, tensor<64xi64>) -> (!HLFHE.eint<6>)
    return %1: !HLFHE.eint<6>
}
)XXX";
  ASSERT_FALSE(engine.compile(mlirStr));
  uint64_t expected = 5;
  for (auto i = 0; i < 100; i++) {
    auto maybeResult = engine.run({expected});
    ASSERT_TRUE((bool)maybeResult);
    uint64_t result = maybeResult.get();
    ASSERT_EQ(result, expected);
  }
}

TEST(CompileAndRunTLU, random_func) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%arg0: !HLFHE.eint<6>) -> !HLFHE.eint<6> {
    %tlu = std.constant dense<[27, 7,37,62,57,16,25,33,49,21,30,19,10,50,14,55,46,35,18,61, 9,20,39,22,29,32, 3, 8,51,52,11,64,40,42, 6,36,15,13,58,41,23,26,54,56,45,44, 5,34,59,47,63, 1,17,12,38, 2,43,28,48,60,31,24, 4,53]> : tensor<64xi64>
    %1 = "HLFHE.apply_lookup_table"(%arg0, %tlu): (!HLFHE.eint<6>, tensor<64xi64>) -> (!HLFHE.eint<6>)
    return %1: !HLFHE.eint<6>
}
)XXX";
  ASSERT_FALSE(engine.compile(mlirStr));

  for (auto i = 0; i < 100; i++) {
    // first value
    auto maybeResult = engine.run({5});
    ASSERT_TRUE((bool)maybeResult);
    uint64_t result = maybeResult.get();
    ASSERT_EQ(result, 16);
    // second value
    maybeResult = engine.run({12});
    ASSERT_TRUE((bool)maybeResult);
    result = maybeResult.get();
    ASSERT_EQ(result, 10);
    // edge value low
    maybeResult = engine.run({0});
    ASSERT_TRUE((bool)maybeResult);
    result = maybeResult.get();
    ASSERT_EQ(result, 27);
    // edge value high
    maybeResult = engine.run({63});
    ASSERT_TRUE((bool)maybeResult);
    result = maybeResult.get();
    ASSERT_EQ(result, 53);
  }
}