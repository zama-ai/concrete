// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>
func @glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}> {
  // CHECK-LABEL: return %arg0 : !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>
  return %arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,57,-25}>
}