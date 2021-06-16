// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: error: should have the same GLWE padding parameter
func @add(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, %arg1: !MidLFHE.glwe<{1024,12,64}{1,7,0,50,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-24}> {
  %0 = "MidLFHE.h_add"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, !MidLFHE.glwe<{1024,12,64}{1,7,0,50,-25}>) -> (!MidLFHE.glwe<{1024,12,64}{0,7,0,50,-24}>)
  return %0: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-24}>
}
