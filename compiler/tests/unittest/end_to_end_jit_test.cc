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
func @main(%arg0: !HLFHE.eint<7>, %arg1: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
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
func @main(%arg0: tensor<2x!HLFHE.eint<7>>, %arg1: tensor<2xi8>, %acc: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  %tacc = tensor.from_elements %acc : tensor<1x!HLFHE.eint<7>>
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!HLFHE.eint<7>>, tensor<2xi8>) outs(%tacc : tensor<1x!HLFHE.eint<7>>) {
  ^bb0(%arg2: !HLFHE.eint<7>, %arg3: i8, %arg4: !HLFHE.eint<7>):  // no predecessors
    %4 = "HLFHE.mul_eint_int"(%arg2, %arg3) : (!HLFHE.eint<7>, i8) -> !HLFHE.eint<7>
    %5 = "HLFHE.add_eint"(%4, %arg4) : (!HLFHE.eint<7>, !HLFHE.eint<7>) -> !HLFHE.eint<7>
    linalg.yield %5 : !HLFHE.eint<7>
  } -> tensor<1x!HLFHE.eint<7>>
  %c0 = constant 0 : index
  %ret = tensor.extract %2[%c0] : tensor<1x!HLFHE.eint<7>>
  return %ret : !HLFHE.eint<7>
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
func @main(%arg0: tensor<4x!HLFHE.eint<7>>,
                   %arg1: tensor<4xi8>) -> !HLFHE.eint<7>
{
  %ret = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<4x!HLFHE.eint<7>>, tensor<4xi8>) -> !HLFHE.eint<7>
  return %ret : !HLFHE.eint<7>
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

TEST(CompileAndRunTLU, tlu) {
  mlir::zamalang::CompilerEngine engine;
  auto mlirStr = R"XXX(
func @main(%arg0: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
    %tlu = std.constant dense<[0, 36028797018963968, 72057594037927936, 108086391056891904, 144115188075855872, 180143985094819840, 216172782113783808, 252201579132747776, 288230376151711744, 324259173170675712, 360287970189639680, 396316767208603648, 432345564227567616, 468374361246531584, 504403158265495552, 540431955284459520, 576460752303423488, 612489549322387456, 648518346341351424, 684547143360315392, 720575940379279360, 756604737398243328, 792633534417207296, 828662331436171264, 864691128455135232, 900719925474099200, 936748722493063168, 972777519512027136, 1008806316530991104, 1044835113549955072, 1080863910568919040, 1116892707587883008, 1152921504606846976, 1188950301625810944, 1224979098644774912, 1261007895663738880, 1297036692682702848, 1333065489701666816, 1369094286720630784, 1405123083739594752, 1441151880758558720, 1477180677777522688, 1513209474796486656, 1549238271815450624, 1585267068834414592, 1621295865853378560, 1657324662872342528, 1693353459891306496, 1729382256910270464, 1765411053929234432, 1801439850948198400, 1837468647967162368, 1873497444986126336, 1909526242005090304, 1945555039024054272, 1981583836043018240, 2017612633061982208, 2053641430080946176, 2089670227099910144, 2125699024118874112, 2161727821137838080, 2197756618156802048, 2233785415175766016, 2269814212194729984, 2305843009213693952, 2341871806232657920, 2377900603251621888, 2413929400270585856, 2449958197289549824, 2485986994308513792, 2522015791327477760, 2558044588346441728, 2594073385365405696, 2630102182384369664, 2666130979403333632, 2702159776422297600, 2738188573441261568, 2774217370460225536, 2810246167479189504, 2846274964498153472, 2882303761517117440, 2918332558536081408, 2954361355555045376, 2990390152574009344, 3026418949592973312, 3062447746611937280, 3098476543630901248, 3134505340649865216, 3170534137668829184, 3206562934687793152, 3242591731706757120, 3278620528725721088, 3314649325744685056, 3350678122763649024, 3386706919782612992, 3422735716801576960, 3458764513820540928, 3494793310839504896, 3530822107858468864, 3566850904877432832, 3602879701896396800, 3638908498915360768, 3674937295934324736, 3710966092953288704, 3746994889972252672, 3783023686991216640, 3819052484010180608, 3855081281029144576, 3891110078048108544, 3927138875067072512, 3963167672086036480, 3999196469105000448, 4035225266123964416, 4071254063142928384, 4107282860161892352, 4143311657180856320, 4179340454199820288, 4215369251218784256, 4251398048237748224, 4287426845256712192, 4323455642275676160, 4359484439294640128, 4395513236313604096, 4431542033332568064, 4467570830351532032, 4503599627370496000, 4539628424389459968, 4575657221408423936]> : tensor<128xi64>
    %1 = "HLFHE.apply_lookup_table"(%arg0, %tlu): (!HLFHE.eint<7>, tensor<128xi64>) -> (!HLFHE.eint<7>)
    return %1: !HLFHE.eint<7>
}
)XXX";
  ASSERT_FALSE(engine.compile(mlirStr));
  auto maybeResult = engine.run({5});
  ASSERT_TRUE((bool)maybeResult);
  uint64_t result = maybeResult.get();
  ASSERT_EQ(result, 5);
}