
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "concretelang/TestLib/TestProgram.h"
#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"

TEST(CompileAndRunClear, add_u64) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%arg0: i64, %arg1: i64) -> i64 {
  %1 = arith.addi %arg0, %arg1 : i64
  return %1: i64
}
)XXX",
             "main", true);
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };
  ASSERT_EQ(lambda({Tensor<uint64_t>(1), Tensor<uint64_t>(2)}), (uint64_t)3);
  ASSERT_EQ(lambda({Tensor<uint64_t>(4), Tensor<uint64_t>(5)}), (uint64_t)9);
  ASSERT_EQ(lambda({Tensor<uint64_t>(1), Tensor<uint64_t>(1)}), (uint64_t)2);
}

TEST(CompileAndRunTensorEncrypted, extract_5) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%t: tensor<10x!FHE.eint<5>>, %i: index) -> !FHE.eint<5>{
  %c = tensor.extract %t[%i] : tensor<10x!FHE.eint<5>>
  return %c : !FHE.eint<5>
}
)XXX");
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };
  Tensor<uint64_t> t_arg({32, 0, 10, 25, 14, 25, 18, 28, 14, 7}, {10});
  for (size_t i = 0; i < 10; i++)
    ASSERT_EQ(lambda({t_arg, Tensor<uint64_t>(i)}), t_arg[i]);
}

TEST(CompileAndRunTensorEncrypted, extract_twice_and_add_5) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%t: tensor<10x!FHE.eint<5>>, %i: index, %j: index) ->
!FHE.eint<5>{
  %ti = tensor.extract %t[%i] : tensor<10x!FHE.eint<5>>
  %tj = tensor.extract %t[%j] : tensor<10x!FHE.eint<5>>
  %c = "FHE.add_eint"(%ti, %tj) : (!FHE.eint<5>, !FHE.eint<5>) ->
  !FHE.eint<5> return %c : !FHE.eint<5>
}
)XXX");
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };
  Tensor<uint64_t> t_arg({3, 0, 7, 12, 14, 6, 5, 4, 1, 2}, {10});
  for (size_t i = 0; i < 10; i++)
    for (size_t j = 0; j < 10; j++)
      ASSERT_EQ(lambda({t_arg, Tensor<uint64_t>(i), Tensor<uint64_t>(j)}),
                t_arg[i] + t_arg[j]);
}

TEST(CompileAndRunTensorEncrypted, dim_5) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%t: tensor<10x!FHE.eint<5>>) -> index{
  %c0 = arith.constant 0 : index
  %c = tensor.dim %t, %c0 : tensor<10x!FHE.eint<5>>
  return %c : index
}
)XXX");
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };
  Tensor<uint64_t> t_arg({32, 0, 10, 25, 14, 25, 18, 28, 14, 7}, {10});
  ASSERT_EQ(lambda({
                t_arg,
            }),
            10_u64);
}

TEST(CompileAndRunTensorEncrypted, from_elements_5) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%0: !FHE.eint<5>) -> tensor<1x!FHE.eint<5>> {
  %t = tensor.from_elements %0 : tensor<1x!FHE.eint<5>>
  return %t: tensor<1x!FHE.eint<5>>
}
)XXX");
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value();
  };
  Tensor<uint64_t> res = lambda({Tensor<uint64_t>(10)});
  ASSERT_EQ(res.values.size(), (size_t)1);
  ASSERT_EQ(res.values[0], 10_u64);
}

TEST(CompileAndRunTensorEncrypted, from_elements_multiple_values) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%0: !FHE.eint<5>, %1: !FHE.eint<5>, %2: !FHE.eint<5>) -> tensor<3x!FHE.eint<5>> {
  %t = tensor.from_elements %0, %1, %2 : tensor<3x!FHE.eint<5>>
  return %t: tensor<3x!FHE.eint<5>>
}
)XXX");
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value();
  };
  Tensor<uint64_t> res =
      lambda({Tensor<uint64_t>(1), Tensor<uint64_t>(2), Tensor<uint64_t>(3)});
  ASSERT_EQ(res.values.size(), (size_t)3);
  ASSERT_EQ(res.values[0], 1_u64);
  ASSERT_EQ(res.values[1], 2_u64);
  ASSERT_EQ(res.values[2], 3_u64);
}

TEST(CompileAndRunTensorEncrypted, from_elements_many_values) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%0: !FHE.eint<5>,
           %1: !FHE.eint<5>,
           %2: !FHE.eint<5>,
           %3: !FHE.eint<5>,
           %4: !FHE.eint<5>,
           %5: !FHE.eint<5>,
           %6: !FHE.eint<5>,
           %7: !FHE.eint<5>,
           %8: !FHE.eint<5>,
           %9: !FHE.eint<5>,
           %10: !FHE.eint<5>,
           %11: !FHE.eint<5>,
           %12: !FHE.eint<5>,
           %13: !FHE.eint<5>,
           %14: !FHE.eint<5>,
           %15: !FHE.eint<5>,
           %16: !FHE.eint<5>,
           %17: !FHE.eint<5>,
           %18: !FHE.eint<5>,
           %19: !FHE.eint<5>,
           %20: !FHE.eint<5>,
           %21: !FHE.eint<5>,
           %22: !FHE.eint<5>,
           %23: !FHE.eint<5>,
           %24: !FHE.eint<5>,
           %25: !FHE.eint<5>,
           %26: !FHE.eint<5>,
           %27: !FHE.eint<5>,
           %28: !FHE.eint<5>,
           %29: !FHE.eint<5>,
           %30: !FHE.eint<5>,
           %31: !FHE.eint<5>,
           %32: !FHE.eint<5>,
           %33: !FHE.eint<5>,
           %34: !FHE.eint<5>,
           %35: !FHE.eint<5>,
           %36: !FHE.eint<5>,
           %37: !FHE.eint<5>,
           %38: !FHE.eint<5>,
           %39: !FHE.eint<5>,
           %40: !FHE.eint<5>,
           %41: !FHE.eint<5>,
           %42: !FHE.eint<5>,
           %43: !FHE.eint<5>,
           %44: !FHE.eint<5>,
           %45: !FHE.eint<5>,
           %46: !FHE.eint<5>,
           %47: !FHE.eint<5>,
           %48: !FHE.eint<5>,
           %49: !FHE.eint<5>,
           %50: !FHE.eint<5>,
           %51: !FHE.eint<5>,
           %52: !FHE.eint<5>,
           %53: !FHE.eint<5>,
           %54: !FHE.eint<5>,
           %55: !FHE.eint<5>,
           %56: !FHE.eint<5>,
           %57: !FHE.eint<5>,
           %58: !FHE.eint<5>,
           %59: !FHE.eint<5>,
           %60: !FHE.eint<5>,
           %61: !FHE.eint<5>,
           %62: !FHE.eint<5>,
           %63: !FHE.eint<5>
) -> tensor<64x!FHE.eint<5>> {
  %t = tensor.from_elements %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63 : tensor<64x!FHE.eint<5>>
  return %t: tensor<64x!FHE.eint<5>>
}
)XXX");
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value();
  };
  Tensor<uint64_t> res = lambda({
      Tensor<uint64_t>(0),  Tensor<uint64_t>(1),  Tensor<uint64_t>(2),
      Tensor<uint64_t>(3),  Tensor<uint64_t>(4),  Tensor<uint64_t>(5),
      Tensor<uint64_t>(6),  Tensor<uint64_t>(7),  Tensor<uint64_t>(8),
      Tensor<uint64_t>(9),  Tensor<uint64_t>(10), Tensor<uint64_t>(11),
      Tensor<uint64_t>(12), Tensor<uint64_t>(13), Tensor<uint64_t>(14),
      Tensor<uint64_t>(15), Tensor<uint64_t>(16), Tensor<uint64_t>(17),
      Tensor<uint64_t>(18), Tensor<uint64_t>(19), Tensor<uint64_t>(20),
      Tensor<uint64_t>(21), Tensor<uint64_t>(22), Tensor<uint64_t>(23),
      Tensor<uint64_t>(24), Tensor<uint64_t>(25), Tensor<uint64_t>(26),
      Tensor<uint64_t>(27), Tensor<uint64_t>(28), Tensor<uint64_t>(29),
      Tensor<uint64_t>(30), Tensor<uint64_t>(31), Tensor<uint64_t>(32),
      Tensor<uint64_t>(33), Tensor<uint64_t>(34), Tensor<uint64_t>(35),
      Tensor<uint64_t>(36), Tensor<uint64_t>(37), Tensor<uint64_t>(38),
      Tensor<uint64_t>(39), Tensor<uint64_t>(40), Tensor<uint64_t>(41),
      Tensor<uint64_t>(42), Tensor<uint64_t>(43), Tensor<uint64_t>(44),
      Tensor<uint64_t>(45), Tensor<uint64_t>(46), Tensor<uint64_t>(47),
      Tensor<uint64_t>(48), Tensor<uint64_t>(49), Tensor<uint64_t>(50),
      Tensor<uint64_t>(51), Tensor<uint64_t>(52), Tensor<uint64_t>(53),
      Tensor<uint64_t>(54), Tensor<uint64_t>(55), Tensor<uint64_t>(56),
      Tensor<uint64_t>(57), Tensor<uint64_t>(58), Tensor<uint64_t>(59),
      Tensor<uint64_t>(60), Tensor<uint64_t>(61), Tensor<uint64_t>(62),
      Tensor<uint64_t>(63),
  });

  ASSERT_EQ(res.values.size(), (size_t)64);
  ASSERT_EQ(res.values[0], 0_u64);
  ASSERT_EQ(res.values[1], 1_u64);
  ASSERT_EQ(res.values[2], 2_u64);
  ASSERT_EQ(res.values[3], 3_u64);
  ASSERT_EQ(res.values[4], 4_u64);
  ASSERT_EQ(res.values[5], 5_u64);
  ASSERT_EQ(res.values[6], 6_u64);
  ASSERT_EQ(res.values[7], 7_u64);
  ASSERT_EQ(res.values[8], 8_u64);
  ASSERT_EQ(res.values[9], 9_u64);
  ASSERT_EQ(res.values[10], 10_u64);
  ASSERT_EQ(res.values[11], 11_u64);
  ASSERT_EQ(res.values[12], 12_u64);
  ASSERT_EQ(res.values[13], 13_u64);
  ASSERT_EQ(res.values[14], 14_u64);
  ASSERT_EQ(res.values[15], 15_u64);
  ASSERT_EQ(res.values[16], 16_u64);
  ASSERT_EQ(res.values[17], 17_u64);
  ASSERT_EQ(res.values[18], 18_u64);
  ASSERT_EQ(res.values[19], 19_u64);
  ASSERT_EQ(res.values[20], 20_u64);
  ASSERT_EQ(res.values[21], 21_u64);
  ASSERT_EQ(res.values[22], 22_u64);
  ASSERT_EQ(res.values[23], 23_u64);
  ASSERT_EQ(res.values[24], 24_u64);
  ASSERT_EQ(res.values[25], 25_u64);
  ASSERT_EQ(res.values[26], 26_u64);
  ASSERT_EQ(res.values[27], 27_u64);
  ASSERT_EQ(res.values[28], 28_u64);
  ASSERT_EQ(res.values[29], 29_u64);
  ASSERT_EQ(res.values[30], 30_u64);
  ASSERT_EQ(res.values[31], 31_u64);
  ASSERT_EQ(res.values[32], 32_u64);
  ASSERT_EQ(res.values[33], 33_u64);
  ASSERT_EQ(res.values[34], 34_u64);
  ASSERT_EQ(res.values[35], 35_u64);
  ASSERT_EQ(res.values[36], 36_u64);
  ASSERT_EQ(res.values[37], 37_u64);
  ASSERT_EQ(res.values[38], 38_u64);
  ASSERT_EQ(res.values[39], 39_u64);
  ASSERT_EQ(res.values[40], 40_u64);
  ASSERT_EQ(res.values[41], 41_u64);
  ASSERT_EQ(res.values[42], 42_u64);
  ASSERT_EQ(res.values[43], 43_u64);
  ASSERT_EQ(res.values[44], 44_u64);
  ASSERT_EQ(res.values[45], 45_u64);
  ASSERT_EQ(res.values[46], 46_u64);
  ASSERT_EQ(res.values[47], 47_u64);
  ASSERT_EQ(res.values[48], 48_u64);
  ASSERT_EQ(res.values[49], 49_u64);
  ASSERT_EQ(res.values[50], 50_u64);
  ASSERT_EQ(res.values[51], 51_u64);
  ASSERT_EQ(res.values[52], 52_u64);
  ASSERT_EQ(res.values[53], 53_u64);
  ASSERT_EQ(res.values[54], 54_u64);
  ASSERT_EQ(res.values[55], 55_u64);
  ASSERT_EQ(res.values[56], 56_u64);
  ASSERT_EQ(res.values[57], 57_u64);
  ASSERT_EQ(res.values[58], 58_u64);
  ASSERT_EQ(res.values[59], 59_u64);
  ASSERT_EQ(res.values[60], 60_u64);
  ASSERT_EQ(res.values[61], 61_u64);
  ASSERT_EQ(res.values[62], 62_u64);
  ASSERT_EQ(res.values[63], 63_u64);
}

TEST(CompileAndRunTensorEncrypted, in_out_tensor_with_op_5) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%in: tensor<2x!FHE.eint<5>>) -> tensor<3x!FHE.eint<5>> {
  %c_0 = arith.constant 0 : index
  %c_1 = arith.constant 1 : index
  %a = tensor.extract %in[%c_0] : tensor<2x!FHE.eint<5>>
  %b = tensor.extract %in[%c_1] : tensor<2x!FHE.eint<5>>
  %aplusa = "FHE.add_eint"(%a, %a): (!FHE.eint<5>, !FHE.eint<5>) ->
  (!FHE.eint<5>) %aplusb = "FHE.add_eint"(%a, %b): (!FHE.eint<5>,
  !FHE.eint<5>) -> (!FHE.eint<5>) %bplusb = "FHE.add_eint"(%b, %b):
  (!FHE.eint<5>, !FHE.eint<5>) -> (!FHE.eint<5>) %out =
  tensor.from_elements %aplusa, %aplusb, %bplusb : tensor<3x!FHE.eint<5>>
  return %out: tensor<3x!FHE.eint<5>>
}
)XXX");
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value();
  };

  Tensor<uint64_t> in({2, 16}, {2});
  Tensor<uint64_t> res = lambda({in});
  ASSERT_EQ(res.values.size(), (size_t)3);
  ASSERT_EQ(res.values[0], (uint64_t)(in[0] + in[0]));
  ASSERT_EQ(res.values[1], (uint64_t)(in[0] + in[1]));
  ASSERT_EQ(res.values[2], (uint64_t)(in[1] + in[1]));
}

// Test is failing since with the bufferization and the parallel options.
// DISABLED as is a bit artificial test, let's investigate later.
TEST(CompileAndRunTensorEncrypted, DISABLED_linalg_generic) {
  checkedJit(testCircuit, R"XXX(
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
func.func @main(%arg0: tensor<2x!FHE.eint<7>>, %arg1: tensor<2xi8>, %acc:
!FHE.eint<7>) -> !FHE.eint<7> {
  %tacc = tensor.from_elements %acc : tensor<1x!FHE.eint<7>>
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types
  = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!FHE.eint<7>>, tensor<2xi8>)
  outs(%tacc : tensor<1x!FHE.eint<7>>) { ^bb0(%arg2: !FHE.eint<7>, %arg3:
  i8, %arg4: !FHE.eint<7>):  // no predecessors
    %4 = "FHE.mul_eint_int"(%arg2, %arg3) : (!FHE.eint<7>, i8) ->
    !FHE.eint<7> %5 = "FHE.add_eint"(%4, %arg4) : (!FHE.eint<7>,
    !FHE.eint<7>) -> !FHE.eint<7> linalg.yield %5 : !FHE.eint<7>
  } -> tensor<1x!FHE.eint<7>>
  %c0 = arith.constant 0 : index
  %ret = tensor.extract %2[%c0] : tensor<1x!FHE.eint<7>>
  return %ret : !FHE.eint<7>
}
)XXX",
             "main", true);
  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };

  Tensor<uint64_t> arg0({2, 8}, {2});
  Tensor<uint8_t> arg1({6, 8}, {2});
  Tensor<uint64_t> acc(0);

  ASSERT_EQ(lambda({arg0, arg1, acc}), 76_u64);
}

TEST(CompileAndRunComposed, compose_add_eint) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  %cst_1 = arith.constant 1 : i4
  %cst_2 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
  %1 = "FHE.add_eint_int"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  %2 = "FHE.apply_lookup_table"(%1, %cst_2): (!FHE.eint<3>, tensor<8xi64>) -> (!FHE.eint<3>)
  return %2: !FHE.eint<3>
}
)XXX",
             "main", false, DEFAULT_dataflowParallelize,
             DEFAULT_loopParallelize, DEFAULT_batchTFHEOps,
             DEFAULT_global_p_error, DEFAULT_chunkedIntegers, DEFAULT_chunkSize,
             DEFAULT_chunkWidth, true);
  auto lambda = [&](std::vector<concretelang::values::Value> args, size_t n) {
    return testCircuit.compose_n_times(args, n)
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };
  ASSERT_EQ(lambda({Tensor<uint64_t>(0)}, 1), (uint64_t)1);
  ASSERT_EQ(lambda({Tensor<uint64_t>(0)}, 2), (uint64_t)2);
  ASSERT_EQ(lambda({Tensor<uint64_t>(0)}, 3), (uint64_t)3);
  ASSERT_EQ(lambda({Tensor<uint64_t>(0)}, 4), (uint64_t)4);
  ASSERT_EQ(lambda({Tensor<uint64_t>(0)}, 5), (uint64_t)5);
  ASSERT_EQ(lambda({Tensor<uint64_t>(0)}, 6), (uint64_t)6);
  ASSERT_EQ(lambda({Tensor<uint64_t>(0)}, 7), (uint64_t)7);
  ASSERT_EQ(lambda({Tensor<uint64_t>(0)}, 8), (uint64_t)0);
}

TEST(CompileNotComposable, not_composable_1) {
  mlir::concretelang::CompilationOptions options;
  options.optimizerConfig.composable = true;
  options.optimizerConfig.strategy = mlir::concretelang::optimizer::DAG_MULTI;
  TestProgram circuit(options);
  auto err = circuit.compile(R"XXX(
func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  %cst_1 = arith.constant 1 : i4
  %1 = "FHE.add_eint_int"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  return %1: !FHE.eint<3>
}
)XXX");
  ASSERT_OUTCOME_HAS_FAILURE_WITH_ERRORMSG(
      err, "Program can not be composed: No luts in the circuit.");
}

TEST(CompileNotComposable, not_composable_2) {
  mlir::concretelang::CompilationOptions options;
  options.optimizerConfig.composable = true;
  options.optimizerConfig.display = true;
  options.optimizerConfig.strategy = mlir::concretelang::optimizer::DAG_MULTI;
  TestProgram circuit(options);
  auto err = circuit.compile(R"XXX(
func.func @main(%arg0: !FHE.eint<3>) -> (!FHE.eint<3>, !FHE.eint<3>) {
  %cst_1 = arith.constant 2 : i4
  %cst_2 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
  %1 = "FHE.mul_eint_int"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  %2 = "FHE.apply_lookup_table"(%1, %cst_2): (!FHE.eint<3>, tensor<8xi64>) -> (!FHE.eint<3>)
  return %1, %2: !FHE.eint<3>, !FHE.eint<3>
}
)XXX");
  ASSERT_OUTCOME_HAS_FAILURE_WITH_ERRORMSG(
      err, "Program can not be composed: Dag is not composable, because of "
           "output 1: Partition 0 has input coefficient 4");
}

TEST(CompileComposable, composable_supported_dag_mono) {
  mlir::concretelang::CompilationOptions options;
  options.optimizerConfig.composable = true;
  options.optimizerConfig.display = true;
  options.optimizerConfig.strategy = mlir::concretelang::optimizer::DAG_MONO;
  TestProgram circuit(options);
  auto err = circuit.compile(R"XXX(
func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  %cst_1 = arith.constant 1 : i4
  %1 = "FHE.add_eint_int"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  return %1: !FHE.eint<3>
}
)XXX");
  assert(err.has_value());
}

TEST(CompileComposable, composable_supported_v0) {
  mlir::concretelang::CompilationOptions options;
  options.optimizerConfig.composable = true;
  options.optimizerConfig.display = true;
  options.optimizerConfig.strategy = mlir::concretelang::optimizer::V0;
  TestProgram circuit(options);
  auto err = circuit.compile(R"XXX(
func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  %cst_1 = arith.constant 1 : i4
  %1 = "FHE.add_eint_int"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  return %1: !FHE.eint<3>
}
)XXX");
  assert(err.has_value());
}

TEST(CompileMultiFunctions, multi_functions_v0) {
  mlir::concretelang::CompilationOptions options;
  options.optimizerConfig.strategy = mlir::concretelang::optimizer::V0;
  TestProgram circuit(options);
  auto err = circuit.compile(R"XXX(
func.func @inc(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  %cst_1 = arith.constant 1 : i4
  %1 = "FHE.add_eint_int"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  return %1: !FHE.eint<3>
}
func.func @dec(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  %cst_1 = arith.constant 1 : i4
  %1 = "FHE.sub_eint_int"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  return %1: !FHE.eint<3>
}
)XXX");
  assert(err.has_value());
  assert(circuit.generateKeyset().has_value());
  auto lambda_inc = [&](std::vector<concretelang::values::Value> args) {
    return circuit.call(args, "inc")
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };
  auto lambda_dec = [&](std::vector<concretelang::values::Value> args) {
    return circuit.call(args, "dec")
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };
  ASSERT_EQ(lambda_inc({Tensor<uint64_t>(1)}), (uint64_t)2);
  ASSERT_EQ(lambda_inc({Tensor<uint64_t>(4)}), (uint64_t)5);
  ASSERT_EQ(lambda_dec({Tensor<uint64_t>(1)}), (uint64_t)0);
  ASSERT_EQ(lambda_dec({Tensor<uint64_t>(4)}), (uint64_t)3);
}

/// https://github.com/zama-ai/concrete-internal/issues/655
TEST(CompileAndRun, compress_input_and_simulate) {
  mlir::concretelang::CompilationOptions options;
  options.compressInputCiphertexts = true;
  options.simulate = true;
  TestProgram circuit(options);
  ASSERT_OUTCOME_HAS_VALUE(circuit.compile(R"XXX(
func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
  return %arg0: !FHE.eint<3>
}
)XXX"));
  ASSERT_ASSIGN_OUTCOME_VALUE(result, circuit.call({Tensor<uint64_t>(7)}));
  ASSERT_EQ(result[0].getTensor<uint64_t>().value()[0], (uint64_t)(7));
}
