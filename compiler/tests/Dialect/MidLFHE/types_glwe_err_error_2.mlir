// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: error: GLWE padding + message + phantom = 25 cannot be represented in bits - scalingFactor = 24
func @glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{15,7,3,40,-25}>) -> !MidLFHE.glwe<{1024,12,64}{15,7,3,40,-25}> {
  return %arg0: !MidLFHE.glwe<{1024,12,64}{15,7,3,40,-25}>
}