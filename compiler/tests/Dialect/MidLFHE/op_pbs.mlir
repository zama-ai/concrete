// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @pbs_ciphertext(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>, %arg1: i32) -> !MidLFHE.glwe<{2048,10,64}{0,7,0,2,-82}> {
func @pbs_ciphertext(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>, %arg1: i32) -> !MidLFHE.glwe<{2048,10,64}{0,7,0,2,-82}> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.pbs"(%arg0) ( {
  // CHECK-NEXT: ^bb0(%[[V2:.*]]: i32):  // no predecessors
  // CHECK-NEXT:   %[[V4:.*]] = divi_unsigned %[[V2]], %arg1 : i32
  // CHECK-NEXT:   "MidLFHE.pbs_return"(%[[V4]]) : (i32) -> ()
  // CHECK-NEXT: }) {base_log = 8 : i32, big_n = 1024 : i32, level = 2 : i32, log_noise = -82 : i32} : (!MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>) -> !MidLFHE.glwe<{2048,10,64}{0,7,0,2,-82}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{2048,10,64}{0,7,0,2,-82}>
  %0 = "MidLFHE.pbs"(%arg0)({
    ^bb0(%a:i32):
      %1 = std.divi_unsigned %a, %arg1 : i32
      "MidLFHE.pbs_return"(%1) : (i32) -> ()
  }){big_n=1024: i32, log_noise=-82: i32, base_log=8 : i32, level=2 : i32} : (!MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>) -> !MidLFHE.glwe<{2048,10,64}{0,7,0,2,-82}>

  return %0 : !MidLFHE.glwe<{2048,10,64}{0,7,0,2,-82}>
}