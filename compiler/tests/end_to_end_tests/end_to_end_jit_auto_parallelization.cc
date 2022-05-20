
#include <concretelang/Runtime/DFRuntime.hpp>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>

#include "end_to_end_jit_test.h"
#include "tests_tools/GtestEnvironment.h"

///////////////////////////////////////////////////////////////////////////////
// Auto-parallelize independent FHE ops /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(ParallelizeAndRunFHE, add_eint_tree) {
  checkedJit(lambda, R"XXX(
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
             "main", false, true, false);

  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    llvm::Expected<uint64_t> res_1 = lambda(1_u64, 2_u64, 3_u64, 4_u64);
    llvm::Expected<uint64_t> res_2 = lambda(4_u64, 5_u64, 6_u64, 7_u64);
    llvm::Expected<uint64_t> res_3 = lambda(1_u64, 1_u64, 1_u64, 1_u64);
    llvm::Expected<uint64_t> res_4 = lambda(5_u64, 7_u64, 11_u64, 13_u64);
    ASSERT_EXPECTED_SUCCESS(res_1);
    ASSERT_EXPECTED_SUCCESS(res_2);
    ASSERT_EXPECTED_SUCCESS(res_3);
    ASSERT_EXPECTED_SUCCESS(res_4);
    ASSERT_EXPECTED_VALUE(res_1, 150);
    ASSERT_EXPECTED_VALUE(res_2, 74);
    ASSERT_EXPECTED_VALUE(res_3, 60);
    ASSERT_EXPECTED_VALUE(res_4, 28);
  } else {
    ASSERT_EXPECTED_FAILURE(lambda());
    ASSERT_EXPECTED_FAILURE(lambda());
    ASSERT_EXPECTED_FAILURE(lambda());
    ASSERT_EXPECTED_FAILURE(lambda());
  }
}

std::vector<uint64_t> parallel_results;

TEST(ParallelizeAndRunFHE, nn_small_parallel) {
  checkedJit(lambda, R"XXX(
  func @main(%arg0: tensor<4x5x!FHE.eint<5>>) -> tensor<4x7x!FHE.eint<5>> {
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

  const size_t numDim = 2;
  const size_t dim0 = 4;
  const size_t dim1 = 5;
  const size_t dim2 = 7;
  const int64_t dims[numDim]{dim0, dim1};
  const llvm::ArrayRef<int64_t> shape2D(dims, numDim);
  std::vector<uint8_t> input;
  input.reserve(dim0 * dim1);

  for (int i = 0; i < dim0 * dim1; ++i)
    input.push_back(i % 17 % 4);

  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint8_t>>
      arg(input, shape2D);

  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    llvm::Expected<std::vector<uint64_t>> res =
        lambda.operator()<std::vector<uint64_t>>({&arg});
    ASSERT_EXPECTED_SUCCESS(res);
    ASSERT_EQ(res->size(), dim0 * dim2);
    parallel_results = *res;
  } else {
    ASSERT_EXPECTED_FAILURE(lambda.operator()<std::vector<uint64_t>>());
  }
}

TEST(ParallelizeAndRunFHE, nn_small_sequential) {
  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    checkedJit(lambda, R"XXX(
  func @main(%arg0: tensor<4x5x!FHE.eint<5>>) -> tensor<4x7x!FHE.eint<5>> {
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

    const size_t numDim = 2;
    const size_t dim0 = 4;
    const size_t dim1 = 5;
    const size_t dim2 = 7;
    const int64_t dims[numDim]{dim0, dim1};
    const llvm::ArrayRef<int64_t> shape2D(dims, numDim);
    std::vector<uint8_t> input;
    input.reserve(dim0 * dim1);

    for (int i = 0; i < dim0 * dim1; ++i)
      input.push_back(i % 17 % 4);

    mlir::concretelang::TensorLambdaArgument<
        mlir::concretelang::IntLambdaArgument<uint8_t>>
        arg(input, shape2D);

    // This is sequential: only execute on root node.
    if (mlir::concretelang::dfr::_dfr_is_root_node()) {
      llvm::Expected<std::vector<uint64_t>> res =
          lambda.operator()<std::vector<uint64_t>>({&arg});
      ASSERT_EXPECTED_SUCCESS(res);
      for (size_t i = 0; i < dim0 * dim2; i++)
        EXPECT_EQ(parallel_results[i], (*res)[i])
            << "result differ at pos " << i;
    }
  }
}
