// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: should have the same GLWE phantomBits parameter
func @add_plain(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,51,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,1,51,-25}> {
  %0 = constant 1 : i32
  %1 = "MidLFHE.add_plain"(%arg0, %0): (!MidLFHE.glwe<{1024,12,64}{0,7,0,51,-25}>, i32) -> (!MidLFHE.glwe<{1024,12,64}{0,7,1,51,-25}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{0,7,1,51,-25}>
}