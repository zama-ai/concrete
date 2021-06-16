// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'MidLFHE.h_add' op has unexpected log2StdDev parameter of its GLWE result, expected:-22
func @add_plain(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>, %arg1: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-23}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-29}> {
  %1 = "MidLFHE.h_add"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>, !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-23}>) -> (!MidLFHE.glwe<{1024,12,64}{0,7,0,57,-29}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-29}>
}