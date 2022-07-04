
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "end_to_end_jit_test.h"

const mlir::concretelang::V0FHEConstraint defaultV0Constraints{10, 7};

TEST(CompileAndRunDFR, start_stop) {
  checkedJit(lambda, R"XXX(
func.func private @_dfr_stop()
func.func private @_dfr_start()
func.func @main() -> i64{
  call @_dfr_start() : () -> ()
  %1 = arith.constant 7 : i64
  call @_dfr_stop() : () -> ()
  return %1 : i64
}
)XXX",
             "main", true);
  ASSERT_EXPECTED_VALUE(lambda(), 7);
}

TEST(CompileAndRunDFR, 0in1out_task) {
  checkedJit(lambda, R"XXX(
  llvm.func @_dfr_await_future(!llvm.ptr<i64>) -> !llvm.ptr<ptr<i64>> attributes {sym_visibility = "private"}
  llvm.func @_dfr_create_async_task(...) attributes {sym_visibility = "private"}
  llvm.func @_dfr_stop()
  llvm.func @_dfr_start()
  func.func @main() -> i64 {
    %0 = llvm.mlir.addressof @_dfr_DFT_work_function__main0 : !llvm.ptr<func<void (ptr<i64>)>>
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(8 : i64) : i64
    llvm.call @_dfr_start() : () -> ()
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%0, %1, %2, %5, %3) : (!llvm.ptr<func<void (ptr<i64>)>>, i64, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %6 = llvm.load %5 : !llvm.ptr<ptr<i64>>
    %7 = llvm.call @_dfr_await_future(%6) : (!llvm.ptr<i64>) -> !llvm.ptr<ptr<i64>>
    %8 = llvm.bitcast %7 : !llvm.ptr<ptr<i64>> to !llvm.ptr<i64>
    %9 = llvm.load %8 : !llvm.ptr<i64>
    llvm.call @_dfr_stop() : () -> ()
    return %9 : i64
  }
  llvm.func @_dfr_DFT_work_function__main0(%arg0: !llvm.ptr<i64>) {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(3 : i64) : i64
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg0 : !llvm.ptr<i64>
    llvm.return
  }
)XXX",
             "main", true);
  ASSERT_EXPECTED_VALUE(lambda(), 7);
}

TEST(CompileAndRunDFR, 1in1out_task) {
  checkedJit(lambda, R"XXX(
  llvm.func @_dfr_await_future(!llvm.ptr<i64>) -> !llvm.ptr<ptr<i64>> attributes {sym_visibility = "private"}
  llvm.func @_dfr_create_async_task(...) attributes {sym_visibility = "private"}
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @_dfr_make_ready_future(...) -> !llvm.ptr<i64> attributes {sym_visibility = "private"}
  llvm.func @_dfr_stop()
  llvm.func @_dfr_start()
  func.func @main(%arg0: i64) -> i64 {
    %0 = llvm.mlir.addressof @_dfr_DFT_work_function__main0 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>)>>
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(8 : i64) : i64
    llvm.call @_dfr_start() : () -> ()
    %3 = llvm.mlir.null : !llvm.ptr<i64>
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.getelementptr %3[%4] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %6 = llvm.ptrtoint %5 : !llvm.ptr<i64> to i64
    %7 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr<i8>
    %8 = llvm.bitcast %7 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %arg0, %8 : !llvm.ptr<i64>
    %9 = llvm.call @_dfr_make_ready_future(%8) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.alloca %10 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%0, %1, %1, %9, %2, %11, %2) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %12 = llvm.load %11 : !llvm.ptr<ptr<i64>>
    %13 = llvm.call @_dfr_await_future(%12) : (!llvm.ptr<i64>) -> !llvm.ptr<ptr<i64>>
    %14 = llvm.bitcast %13 : !llvm.ptr<ptr<i64>> to !llvm.ptr<i64>
    %15 = llvm.load %14 : !llvm.ptr<i64>
    llvm.call @_dfr_stop() : () -> ()
    return %15 : i64
  }
  llvm.func @_dfr_DFT_work_function__main0(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>) {
    %0 = llvm.mlir.constant(2 : i64) : i64
    %1 = llvm.load %arg0 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %1, %0  : i64
    llvm.store %2, %arg1 : !llvm.ptr<i64>
    llvm.return
  }
)XXX",
             "main", true);

  ASSERT_EXPECTED_VALUE(lambda(5_u64), 7);
}

TEST(CompileAndRunDFR, 2in1out_task) {
  checkedJit(lambda, R"XXX(
  llvm.func @_dfr_await_future(!llvm.ptr<i64>) -> !llvm.ptr<ptr<i64>> attributes {sym_visibility = "private"}
  llvm.func @_dfr_create_async_task(...) attributes {sym_visibility = "private"}
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @_dfr_make_ready_future(...) -> !llvm.ptr<i64> attributes {sym_visibility = "private"}
  llvm.func @_dfr_stop()
  llvm.func @_dfr_start()
  func.func @main(%arg0: i64, %arg1: i64) -> i64 {
    %0 = llvm.mlir.addressof @_dfr_DFT_work_function__main0 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(8 : i64) : i64
    llvm.call @_dfr_start() : () -> ()
    %4 = llvm.mlir.null : !llvm.ptr<i64>
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.getelementptr %4[%5] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %7 = llvm.ptrtoint %6 : !llvm.ptr<i64> to i64
    %8 = llvm.call @malloc(%7) : (i64) -> !llvm.ptr<i8>
    %9 = llvm.bitcast %8 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %arg0, %9 : !llvm.ptr<i64>
    %10 = llvm.call @_dfr_make_ready_future(%9) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %11 = llvm.mlir.null : !llvm.ptr<i64>
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.getelementptr %11[%12] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %14 = llvm.ptrtoint %13 : !llvm.ptr<i64> to i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr<i8>
    %16 = llvm.bitcast %15 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %arg1, %16 : !llvm.ptr<i64>
    %17 = llvm.call @_dfr_make_ready_future(%16) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %18 = llvm.mlir.constant(1 : i64) : i64
    %19 = llvm.alloca %18 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%0, %1, %2, %10, %3, %17, %3, %19, %3) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %20 = llvm.load %19 : !llvm.ptr<ptr<i64>>
    %21 = llvm.call @_dfr_await_future(%20) : (!llvm.ptr<i64>) -> !llvm.ptr<ptr<i64>>
    %22 = llvm.bitcast %21 : !llvm.ptr<ptr<i64>> to !llvm.ptr<i64>
    %23 = llvm.load %22 : !llvm.ptr<i64>
    llvm.call @_dfr_stop() : () -> ()
    return %23 : i64
  }
  llvm.func @_dfr_DFT_work_function__main0(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
)XXX",
             "main", true);

  ASSERT_EXPECTED_VALUE(lambda(1_u64, 6_u64), 7);
}

TEST(CompileAndRunDFR, taskgraph) {
  checkedJit(lambda, R"XXX(
  llvm.func @_dfr_await_future(!llvm.ptr<i64>) -> !llvm.ptr<ptr<i64>> attributes {sym_visibility = "private"}
  llvm.func @_dfr_create_async_task(...) attributes {sym_visibility = "private"}
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @_dfr_make_ready_future(...) -> !llvm.ptr<i64> attributes {sym_visibility = "private"}
  llvm.func @_dfr_stop()
  llvm.func @_dfr_start()
  func.func @main(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %0 = llvm.mlir.constant(7 : i64) : i64
    %1 = llvm.mlir.addressof @_dfr_DFT_work_function__main0 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(8 : i64) : i64
    %5 = llvm.mlir.addressof @_dfr_DFT_work_function__main1 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    %6 = llvm.mlir.addressof @_dfr_DFT_work_function__main2 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    %7 = llvm.mlir.addressof @_dfr_DFT_work_function__main3 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    %8 = llvm.mlir.addressof @_dfr_DFT_work_function__main4 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    %9 = llvm.mlir.addressof @_dfr_DFT_work_function__main5 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    %10 = llvm.mlir.addressof @_dfr_DFT_work_function__main6 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    %11 = llvm.mlir.addressof @_dfr_DFT_work_function__main7 : !llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>
    llvm.call @_dfr_start() : () -> ()
    %12 = llvm.mlir.null : !llvm.ptr<i64>
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.getelementptr %12[%13] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %15 = llvm.ptrtoint %14 : !llvm.ptr<i64> to i64
    %16 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr<i8>
    %17 = llvm.bitcast %16 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %arg0, %17 : !llvm.ptr<i64>
    %18 = llvm.call @_dfr_make_ready_future(%17) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %19 = llvm.mlir.null : !llvm.ptr<i64>
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.getelementptr %19[%20] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %22 = llvm.ptrtoint %21 : !llvm.ptr<i64> to i64
    %23 = llvm.call @malloc(%22) : (i64) -> !llvm.ptr<i8>
    %24 = llvm.bitcast %23 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %arg1, %24 : !llvm.ptr<i64>
    %25 = llvm.call @_dfr_make_ready_future(%24) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %26 = llvm.mlir.constant(1 : i64) : i64
    %27 = llvm.alloca %26 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%1, %2, %3, %18, %4, %25, %4, %27, %4) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %28 = llvm.load %27 : !llvm.ptr<ptr<i64>>
    %29 = llvm.mlir.null : !llvm.ptr<i64>
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.getelementptr %29[%30] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %32 = llvm.ptrtoint %31 : !llvm.ptr<i64> to i64
    %33 = llvm.call @malloc(%32) : (i64) -> !llvm.ptr<i8>
    %34 = llvm.bitcast %33 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %arg2, %34 : !llvm.ptr<i64>
    %35 = llvm.call @_dfr_make_ready_future(%34) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %36 = llvm.mlir.constant(1 : i64) : i64
    %37 = llvm.alloca %36 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%5, %2, %3, %18, %4, %35, %4, %37, %4) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %38 = llvm.load %37 : !llvm.ptr<ptr<i64>>
    %39 = llvm.mlir.constant(1 : i64) : i64
    %40 = llvm.alloca %39 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%6, %2, %3, %25, %4, %35, %4, %40, %4) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %41 = llvm.load %40 : !llvm.ptr<ptr<i64>>
    %42 = llvm.mul %arg0, %0  : i64
    %43 = llvm.mul %arg1, %0  : i64
    %44 = llvm.mul %arg2, %0  : i64
    %45 = llvm.mlir.null : !llvm.ptr<i64>
    %46 = llvm.mlir.constant(1 : index) : i64
    %47 = llvm.getelementptr %45[%46] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %48 = llvm.ptrtoint %47 : !llvm.ptr<i64> to i64
    %49 = llvm.call @malloc(%48) : (i64) -> !llvm.ptr<i8>
    %50 = llvm.bitcast %49 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %42, %50 : !llvm.ptr<i64>
    %51 = llvm.call @_dfr_make_ready_future(%50) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %52 = llvm.mlir.constant(1 : i64) : i64
    %53 = llvm.alloca %52 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%7, %2, %3, %28, %4, %51, %4, %53, %4) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %54 = llvm.load %53 : !llvm.ptr<ptr<i64>>
    %55 = llvm.mlir.null : !llvm.ptr<i64>
    %56 = llvm.mlir.constant(1 : index) : i64
    %57 = llvm.getelementptr %55[%56] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %58 = llvm.ptrtoint %57 : !llvm.ptr<i64> to i64
    %59 = llvm.call @malloc(%58) : (i64) -> !llvm.ptr<i8>
    %60 = llvm.bitcast %59 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %43, %60 : !llvm.ptr<i64>
    %61 = llvm.call @_dfr_make_ready_future(%60) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %62 = llvm.mlir.constant(1 : i64) : i64
    %63 = llvm.alloca %62 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%8, %2, %3, %38, %4, %61, %4, %63, %4) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %64 = llvm.load %63 : !llvm.ptr<ptr<i64>>
    %65 = llvm.mlir.null : !llvm.ptr<i64>
    %66 = llvm.mlir.constant(1 : index) : i64
    %67 = llvm.getelementptr %65[%66] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %68 = llvm.ptrtoint %67 : !llvm.ptr<i64> to i64
    %69 = llvm.call @malloc(%68) : (i64) -> !llvm.ptr<i8>
    %70 = llvm.bitcast %69 : !llvm.ptr<i8> to !llvm.ptr<i64>
    llvm.store %44, %70 : !llvm.ptr<i64>
    %71 = llvm.call @_dfr_make_ready_future(%70) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %72 = llvm.mlir.constant(1 : i64) : i64
    %73 = llvm.alloca %72 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%9, %2, %3, %41, %4, %71, %4, %73, %4) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %74 = llvm.load %73 : !llvm.ptr<ptr<i64>>
    %75 = llvm.mlir.constant(1 : i64) : i64
    %76 = llvm.alloca %75 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%10, %2, %3, %54, %4, %64, %4, %76, %4) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %77 = llvm.load %76 : !llvm.ptr<ptr<i64>>
    %78 = llvm.mlir.constant(1 : i64) : i64
    %79 = llvm.alloca %78 x !llvm.ptr<i64> : (i64) -> !llvm.ptr<ptr<i64>>
    llvm.call @_dfr_create_async_task(%11, %2, %3, %77, %4, %74, %4, %79, %4) : (!llvm.ptr<func<void (ptr<i64>, ptr<i64>, ptr<i64>)>>, i64, i64, !llvm.ptr<i64>, i64, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<i64>>, i64) -> ()
    %80 = llvm.load %79 : !llvm.ptr<ptr<i64>>
    %81 = llvm.call @_dfr_await_future(%80) : (!llvm.ptr<i64>) -> !llvm.ptr<ptr<i64>>
    %82 = llvm.bitcast %81 : !llvm.ptr<ptr<i64>> to !llvm.ptr<i64>
    %83 = llvm.load %82 : !llvm.ptr<i64>
    llvm.call @_dfr_stop() : () -> ()
    return %83 : i64
  }
  llvm.func @_dfr_DFT_work_function__main0(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) attributes {_dfr_work_function_attribute} {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
  llvm.func @_dfr_DFT_work_function__main1(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) attributes {_dfr_work_function_attribute} {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
  llvm.func @_dfr_DFT_work_function__main2(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) attributes {_dfr_work_function_attribute} {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
  llvm.func @_dfr_DFT_work_function__main3(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) attributes {_dfr_work_function_attribute} {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
  llvm.func @_dfr_DFT_work_function__main4(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) attributes {_dfr_work_function_attribute} {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
  llvm.func @_dfr_DFT_work_function__main5(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) attributes {_dfr_work_function_attribute} {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
  llvm.func @_dfr_DFT_work_function__main6(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) attributes {_dfr_work_function_attribute} {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
  llvm.func @_dfr_DFT_work_function__main7(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>) attributes {_dfr_work_function_attribute} {
    %0 = llvm.load %arg0 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.add %0, %1  : i64
    llvm.store %2, %arg2 : !llvm.ptr<i64>
    llvm.return
  }
)XXX",
             "main", true);

  ASSERT_EXPECTED_VALUE(lambda(1_u64, 2_u64, 3_u64), 54);
  ASSERT_EXPECTED_VALUE(lambda(2_u64, 5_u64, 1_u64), 72);
  ASSERT_EXPECTED_VALUE(lambda(3_u64, 1_u64, 7_u64), 99);
}
