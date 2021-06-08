// RUN: zamacompiler %s  2>&1| FileCheck %s


// CHECK-LABEL: func @add_plain_glwe(%arg0: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>, %arg1: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>) -> !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
func @add_plain_glwe(%arg0: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>, %arg1: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>) -> !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.h_add"(%arg0, %arg1) : (!MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>, !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>) -> !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>

  %0 = "MidLFHE.h_add"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>, !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>) -> (!MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>)
  return %0: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
}
