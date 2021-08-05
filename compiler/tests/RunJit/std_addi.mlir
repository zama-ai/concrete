// RUN: zamacompiler %s --run-jit --jit-args 11 --jit-args 31 2>&1| FileCheck %s

// CHECK-LABEL: 42
func @main(%arg0: i64, %arg1: i64) -> i64 {
  %c = addi %arg0, %arg1 : i64
  return %c : i64
}