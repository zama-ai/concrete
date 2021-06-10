// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: error: the result shoud have 0 paddingBits has input has 0 paddingBits
func @mul_plain(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>) -> !MidLFHE.glwe<{1024,12,64}{1,7,0,50,-25}> {
  %0 = constant 1 : i32
  %1 = "MidLFHE.mul_plain"(%arg0, %0): (!MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, i32) -> (!MidLFHE.glwe<{1024,12,64}{1,7,0,50,-25}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{1,7,0,50,-25}>
}