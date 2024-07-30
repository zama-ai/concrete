// RUN: concretecompiler --split-input-file --action=dump-std --parallelize --parallelize-loops --skip-program-info --passes=for-loop-to-parallel --skip-program-info %s 2>&1| FileCheck %s

func.func @bar() -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32

  // CHECK-NOT: scf.parallel
  %0 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ia = %i0) -> i32 {
    "Tracing.trace_plaintext"(%i0) : (i32) -> ()
    %yld = arith.addi %ia, %i1 : i32
    scf.yield %yld : i32
  } {"parallel" = false }
  
  return
}
