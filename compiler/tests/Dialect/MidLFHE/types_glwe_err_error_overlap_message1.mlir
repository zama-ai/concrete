// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: error: GLWE error overlap message, errBits(41) > scalingFactor(40) + phantomBits(0)
func @glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,40,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,40,-25}> {
  return %arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,40,-25}>
}