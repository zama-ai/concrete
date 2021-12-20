
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "end_to_end_jit_test.h"

///////////////////////////////////////////////////////////////////////////////
// Auto-parallelize independent HLFHE ops /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(ParallelizeAndRunHLFHE, add_eint_tree) {
  mlir::zamalang::JitCompilerEngine::Lambda lambda = checkedJit(R"XXX(
func @main(%arg0: !HLFHE.eint<7>, %arg1: !HLFHE.eint<7>, %arg2: !HLFHE.eint<7>, %arg3: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %2 = "HLFHE.add_eint"(%arg0, %arg2): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %3 = "HLFHE.add_eint"(%arg0, %arg3): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %4 = "HLFHE.add_eint"(%arg1, %arg2): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %5 = "HLFHE.add_eint"(%arg1, %arg3): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %6 = "HLFHE.add_eint"(%arg2, %arg3): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)

  %7 = "HLFHE.add_eint"(%1, %2): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %8 = "HLFHE.add_eint"(%1, %3): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %9 = "HLFHE.add_eint"(%1, %4): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %10 = "HLFHE.add_eint"(%1, %5): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %11 = "HLFHE.add_eint"(%1, %6): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %12 = "HLFHE.add_eint"(%2, %3): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %13 = "HLFHE.add_eint"(%2, %4): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %14 = "HLFHE.add_eint"(%2, %5): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %15 = "HLFHE.add_eint"(%2, %6): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %16 = "HLFHE.add_eint"(%3, %4): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %17 = "HLFHE.add_eint"(%3, %5): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %18 = "HLFHE.add_eint"(%3, %6): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %19 = "HLFHE.add_eint"(%4, %5): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %20 = "HLFHE.add_eint"(%4, %6): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %21 = "HLFHE.add_eint"(%5, %6): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)

  %22 = "HLFHE.add_eint"(%7, %8): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %23 = "HLFHE.add_eint"(%9, %10): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %24 = "HLFHE.add_eint"(%11, %12): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %25 = "HLFHE.add_eint"(%13, %14): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %26 = "HLFHE.add_eint"(%15, %16): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %27 = "HLFHE.add_eint"(%17, %18): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %28 = "HLFHE.add_eint"(%19, %20): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)

  %29 = "HLFHE.add_eint"(%22, %23): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %30 = "HLFHE.add_eint"(%24, %25): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %31 = "HLFHE.add_eint"(%26, %27): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %32 = "HLFHE.add_eint"(%21, %28): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)

  %33 = "HLFHE.add_eint"(%29, %30): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  %34 = "HLFHE.add_eint"(%31, %32): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)

  %35 = "HLFHE.add_eint"(%33, %34): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %35: !HLFHE.eint<7>
}
)XXX", "main", false, true);

  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2_u64, 3_u64, 4_u64), 150);
  ASSERT_EXPECTED_VALUE(lambda(4_u64, 5_u64, 6_u64, 7_u64), 74);
  ASSERT_EXPECTED_VALUE(lambda(1_u64, 1_u64, 1_u64, 1_u64), 60);
  ASSERT_EXPECTED_VALUE(lambda(5_u64, 7_u64, 11_u64, 13_u64), 28);
}
