
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "end_to_end_jit_test.h"

///////////////////////////////////////////////////////////////////////////////
// Auto-parallelize independent FHE ops /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(ParallelizeAndRunFHE, add_eint_tree) {
  checkedJit(lambda, R"XXX(
func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>, %arg2: !FHE.eint<7>, %arg3: !FHE.eint<7>) -> !FHE.eint<7> {
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
             "main", false, true);

  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2_u64, 3_u64, 4_u64), 150);
  ASSERT_EXPECTED_VALUE(lambda(4_u64, 5_u64, 6_u64, 7_u64), 74);
  ASSERT_EXPECTED_VALUE(lambda(1_u64, 1_u64, 1_u64, 1_u64), 60);
  ASSERT_EXPECTED_VALUE(lambda(5_u64, 7_u64, 11_u64, 13_u64), 28);
}
