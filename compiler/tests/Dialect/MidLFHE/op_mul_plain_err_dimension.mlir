// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: error: should have the same GLWE dimension parameter
func @mul_plain(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>) -> !MidLFHE.glwe<{1023,12,64}{0,7,0,57,-25}> {
  %0 = constant 1 : i32
  %1 = "MidLFHE.mul_plain"(%arg0, %0): (!MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>, i32) -> (!MidLFHE.glwe<{1023,12,64}{0,7,0,57,-25}>)
  return %1: !MidLFHE.glwe<{1023,12,64}{0,7,0,57,-25}>
}
