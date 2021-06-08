// RUN: zamacompiler %s  2>&1| FileCheck %s
// CHECK-LABEL: func @mul_plain_glwe(%arg0: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>) -> !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
func @mul_plain_glwe(%arg0: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>) -> !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.mul_plain"(%arg0, %[[V1]]) : (!MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>, i32) -> !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>

  %0 = constant 1 : i32
  %1 = "MidLFHE.mul_plain"(%arg0, %0): (!MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>, i32) -> (!MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>)
  return %1: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
}