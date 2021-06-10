// RUN: zamacompiler %s  2>&1| FileCheck %s


// CHECK-LABEL: func @mul_plain_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, %arg1: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>
func @mul_plain_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, %arg1: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.h_mul"(%arg0, %arg1) : (!MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>) -> !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>

  %0 = "MidLFHE.h_mul"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>, !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>) -> (!MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>)
  return %0: !MidLFHE.glwe<{1024,12,64}{0,7,0,50,-25}>
}
