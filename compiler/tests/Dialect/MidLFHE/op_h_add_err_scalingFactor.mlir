// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: should have the same GLWE scalingFactor parameter
func @add(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, %arg1: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,49,-25}> {
  %1 = "MidLFHE.h_add"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>) -> (!MidLFHE.glwe<{1024,12,64}{0,7,0,49,-25}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{0,7,0,49,-25}>
}