// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK: func.func @keyswitch_glwe(%[[A0:.*]]: !TFHE.glwe<{1,1024,64}{7}>) -> !TFHE.glwe<{1,527,64}{7}>
func.func @keyswitch_glwe(%arg0: !TFHE.glwe<{1,1024,64}{7}>) -> !TFHE.glwe<{1,527,64}{7}> {
  // CHECK-NEXT: %[[V0:.*]] = "TFHE.keyswitch_glwe"(%[[A0]]) {baseLog = 2 : i32, level = 3 : i32} : (!TFHE.glwe<{1,1024,64}{7}>) -> !TFHE.glwe<{1,527,64}{7}>
  // CHECK-NEXT: return %[[V0]] : !TFHE.glwe<{1,527,64}{7}
  %0 = "TFHE.keyswitch_glwe"(%arg0) {baseLog = 2 : i32, level = 3 : i32} : (!TFHE.glwe<{1,1024,64}{7}>) -> !TFHE.glwe<{1,527,64}{7}>
  return %0: !TFHE.glwe<{1,527,64}{7}>
}

// CHECK: func.func @bootstrap_glwe(%[[GLWE:.*]]: !TFHE.glwe<{1,527,64}{7}>, %[[LUT:.*]]: tensor<128xi64>) -> !TFHE.glwe<{1,1024,64}{7}>
func.func @bootstrap_glwe(%glwe: !TFHE.glwe<{1,527,64}{7}>, %lut: tensor<128xi64>) -> !TFHE.glwe<{1,1024,64}{7}> {
    // CHECK-NEXT: %[[V0:.*]] = "TFHE.bootstrap_glwe"(%[[GLWE]], %[[LUT]]) {baseLog = 2 : i32, glweDimension = 1 : i32, inputLweDim = 527 : i32, level = 3 : i32, polySize = 2048 : i32} : (!TFHE.glwe<{1,527,64}{7}>, tensor<128xi64>) -> !TFHE.glwe<{1,1024,64}{7}>
    // CHECK-NEXT: return %[[V0]] : !TFHE.glwe<{1,1024,64}{7}>
    %0 = "TFHE.bootstrap_glwe"(%glwe, %lut) {baseLog = 2 : i32, glweDimension = 1 : i32, inputLweDim = 527 : i32, level = 3 : i32, polySize = 2048 : i32} : (!TFHE.glwe<{1,527,64}{7}>, tensor<128xi64>) -> !TFHE.glwe<{1,1024,64}{7}>
    return %0 : !TFHE.glwe<{1,1024,64}{7}>
}

