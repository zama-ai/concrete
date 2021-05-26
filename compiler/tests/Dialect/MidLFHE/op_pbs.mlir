// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @pbs_ciphertext(%arg0: !MidLFHE.ciphertext, %arg1: i32) -> !MidLFHE.ciphertext {
func @pbs_ciphertext(%arg0: !MidLFHE.ciphertext, %arg1: i32) -> !MidLFHE.ciphertext {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.pbs"(%arg0) ( {
  // CHECK-NEXT: ^bb0(%[[V2:.*]]: i32):  // no predecessors
  // CHECK-NEXT:   %[[V4:.*]] = divi_unsigned %[[V2]], %arg1 : i32
  // CHECK-NEXT:   "MidLFHE.pbs_return"(%[[V4]]) : (i32) -> ()
  // CHECK-NEXT: }) : (!MidLFHE.ciphertext) -> !MidLFHE.ciphertext
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.ciphertext
  %0 = "MidLFHE.pbs"(%arg0)({
    ^bb0(%a:i32):
      %1 = std.divi_unsigned %a, %arg1 : i32
      "MidLFHE.pbs_return"(%1) : (i32) -> ()
  }) : (!MidLFHE.ciphertext) -> !MidLFHE.ciphertext

  return %0 : !MidLFHE.ciphertext
}