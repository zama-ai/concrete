
#include <concretelang/Runtime/DFRuntime.hpp>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>

#include "concretelang/TestLib/TestProgram.h"
#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"
///////////////////////////////////////////////////////////////////////////////
// Auto-parallelize independent FHE ops /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(ParallelizeAndRunFHE, add_eint_tree) {
  checkedJit(testCircuit, R"XXX(
func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>, %arg2: !FHE.eint<7>, %arg3: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %2 = "FHE.add_eint"(%arg0, %arg2): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %3 = "FHE.add_eint"(%arg0, %arg3): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %4 = "FHE.add_eint"(%arg1, %arg2): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %5 = "FHE.add_eint"(%arg1, %arg3): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %6 = "FHE.add_eint"(%arg2, %arg3): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)

  %7 = "FHE.add_eint"(%1, %2): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %8 = "FHE.add_eint"(%1, %3): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %9 = "FHE.add_eint"(%1, %4): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %10 = "FHE.add_eint"(%1, %5): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %11 = "FHE.add_eint"(%1, %6): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %12 = "FHE.add_eint"(%2, %3): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %13 = "FHE.add_eint"(%2, %4): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %14 = "FHE.add_eint"(%2, %5): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %15 = "FHE.add_eint"(%2, %6): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %16 = "FHE.add_eint"(%3, %4): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %17 = "FHE.add_eint"(%3, %5): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %18 = "FHE.add_eint"(%3, %6): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %19 = "FHE.add_eint"(%4, %5): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %20 = "FHE.add_eint"(%4, %6): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %21 = "FHE.add_eint"(%5, %6): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)

  %22 = "FHE.add_eint"(%7, %8): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %23 = "FHE.add_eint"(%9, %10): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %24 = "FHE.add_eint"(%11, %12): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %25 = "FHE.add_eint"(%13, %14): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %26 = "FHE.add_eint"(%15, %16): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %27 = "FHE.add_eint"(%17, %18): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %28 = "FHE.add_eint"(%19, %20): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)

  %29 = "FHE.add_eint"(%22, %23): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %30 = "FHE.add_eint"(%24, %25): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %31 = "FHE.add_eint"(%26, %27): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %32 = "FHE.add_eint"(%21, %28): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)

  %33 = "FHE.add_eint"(%29, %30): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  %34 = "FHE.add_eint"(%31, %32): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)

  %35 = "FHE.add_eint"(%33, %34): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %35: !FHE.eint<7>
}
)XXX",
             "main", false, true, false, false, 1e-40);

  auto lambda = [&](std::vector<concretelang::values::Value> args) {
    return testCircuit.call(args)
        .value()[0]
        .template getTensor<uint64_t>()
        .value()[0];
  };

  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    ASSERT_EQ(lambda({Tensor<uint64_t>(1), Tensor<uint64_t>(2),
                      Tensor<uint64_t>(3), Tensor<uint64_t>(4)}),
              (uint64_t)150);
    ASSERT_EQ(lambda({Tensor<uint64_t>(4), Tensor<uint64_t>(5),
                      Tensor<uint64_t>(6), Tensor<uint64_t>(7)}),
              (uint64_t)74);
    ASSERT_EQ(lambda({Tensor<uint64_t>(1), Tensor<uint64_t>(1),
                      Tensor<uint64_t>(1), Tensor<uint64_t>(1)}),
              (uint64_t)60);
    ASSERT_EQ(lambda({Tensor<uint64_t>(5), Tensor<uint64_t>(7),
                      Tensor<uint64_t>(11), Tensor<uint64_t>(13)}),
              (uint64_t)28);
  } else {
    ASSERT_OUTCOME_HAS_FAILURE(testCircuit.call({}));
    ASSERT_OUTCOME_HAS_FAILURE(testCircuit.call({}));
    ASSERT_OUTCOME_HAS_FAILURE(testCircuit.call({}));
    ASSERT_OUTCOME_HAS_FAILURE(testCircuit.call({}));
  }
}

std::vector<uint64_t> parallel_results;

TEST(ParallelizeAndRunFHE, nn_small_parallel) {
  checkedJit(lambda, R"XXX(
  func.func @main(%arg0: tensor<4x5x!FHE.eint<5>>) -> tensor<4x7x!FHE.eint<5>> {
    %cst = arith.constant dense<[[0, 0, 1, 0, 1, 1, 0], [1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]]> : tensor<4x7xi6>
    %cst_0 = arith.constant dense<[[1, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 1], [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1]]> : tensor<5x7xi6>
    %0 = "FHELinalg.matmul_eint_int"(%arg0, %cst_0) : (tensor<4x5x!FHE.eint<5>>, tensor<5x7xi6>) -> tensor<4x7x!FHE.eint<5>>
    %1 = "FHELinalg.add_eint_int"(%0, %cst) : (tensor<4x7x!FHE.eint<5>>, tensor<4x7xi6>) -> tensor<4x7x!FHE.eint<5>>
    %cst_1 = arith.constant dense<[0, 3, 7, 10, 14, 17, 21, 24, 28, 31, 35, 38, 42, 45, 49, 52, 56, 59, 63, 66, 70, 73, 77, 80, 84, 87, 91, 94, 98, 101, 105, 108]> : tensor<32xi64>
    %2 = "FHELinalg.apply_lookup_table"(%1, %cst_1) : (tensor<4x7x!FHE.eint<5>>, tensor<32xi64>) -> tensor<4x7x!FHE.eint<5>>
    return %2 : tensor<4x7x!FHE.eint<5>>
  }
)XXX",
             "main", false, true, true);

  const size_t dim0 = 4;
  const size_t dim1 = 5;
  const size_t dim2 = 7;
  const std::vector<size_t> inputShape({dim0, dim1});
  const std::vector<size_t> outputShape({dim0, dim2});
  std::vector<uint64_t> values;
  values.reserve(dim0 * dim1);
  for (size_t i = 0; i < dim0 * dim1; ++i) {
    values.push_back(i % 17 % 4);
  }
  auto input = Tensor<uint64_t>(values, inputShape);

  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    auto maybeResult = lambda.call({input});
    ASSERT_OUTCOME_HAS_VALUE(maybeResult);
    auto result = maybeResult.value()[0].template getTensor<uint64_t>().value();
    ASSERT_EQ(result.dimensions, outputShape);
    parallel_results = result.values;
  } else {
    ASSERT_OUTCOME_HAS_FAILURE(lambda.call({}));
  }
}

TEST(ParallelizeAndRunFHE, nn_small_sequential) {
  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    checkedJit(lambda, R"XXX(
  func.func @main(%arg0: tensor<4x5x!FHE.eint<5>>) -> tensor<4x7x!FHE.eint<5>> {
    %cst = arith.constant dense<[[0, 0, 1, 0, 1, 1, 0], [1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]]> : tensor<4x7xi6>
    %cst_0 = arith.constant dense<[[1, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 1], [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1]]> : tensor<5x7xi6>
    %0 = "FHELinalg.matmul_eint_int"(%arg0, %cst_0) : (tensor<4x5x!FHE.eint<5>>, tensor<5x7xi6>) -> tensor<4x7x!FHE.eint<5>>
    %1 = "FHELinalg.add_eint_int"(%0, %cst) : (tensor<4x7x!FHE.eint<5>>, tensor<4x7xi6>) -> tensor<4x7x!FHE.eint<5>>
    %cst_1 = arith.constant dense<[0, 3, 7, 10, 14, 17, 21, 24, 28, 31, 35, 38, 42, 45, 49, 52, 56, 59, 63, 66, 70, 73, 77, 80, 84, 87, 91, 94, 98, 101, 105, 108]> : tensor<32xi64>
    %2 = "FHELinalg.apply_lookup_table"(%1, %cst_1) : (tensor<4x7x!FHE.eint<5>>, tensor<32xi64>) -> tensor<4x7x!FHE.eint<5>>
    return %2 : tensor<4x7x!FHE.eint<5>>
  }
)XXX",
               "main", false, false, false);

    const size_t dim0 = 4;
    const size_t dim1 = 5;
    const size_t dim2 = 7;
    const std::vector<size_t> inputShape({dim0, dim1});
    const std::vector<size_t> outputShape({dim0, dim2});
    std::vector<uint64_t> values;
    values.reserve(dim0 * dim1);
    for (size_t i = 0; i < dim0 * dim1; ++i) {
      values.push_back(i % 17 % 4);
    }
    auto input = Tensor<uint64_t>(values, inputShape);

    if (mlir::concretelang::dfr::_dfr_is_root_node()) {
      auto maybeResult = lambda.call({input});
      ASSERT_OUTCOME_HAS_VALUE(maybeResult);
      auto result =
          maybeResult.value()[0].template getTensor<uint64_t>().value();
      for (size_t i = 0; i < dim0 * dim2; i++)
        EXPECT_EQ(parallel_results[i], result.values[i])
            << "result differ at pos " << i;
    }
  }
}
