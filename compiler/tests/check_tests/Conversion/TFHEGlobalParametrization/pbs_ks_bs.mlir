// RUN: concretecompiler --passes tfhe-global-parametrization --action=dump-std --optimizer-v0 --v0-parameter=2,10,750,1,23,3,4 --v0-constraint=4,0 %s 2>&1| FileCheck %s

//CHECK: func @main(%[[A0:.*]]: !TFHE.glwe<{2048,1,64}{4}>) -> !TFHE.glwe<{2048,1,64}{4}> {
//CHECK:   %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
//CHECK:   %[[V0:.*]] = "TFHE.glwe_from_table"(%cst) : (tensor<16xi64>) -> !TFHE.glwe<{2,1024,64}{4}>
//CHECK:   %[[V1:.*]] = "TFHE.keyswitch_glwe"(%[[A0]]) {baseLog = 4 : i32, level = 3 : i32} : (!TFHE.glwe<{2048,1,64}{4}>) -> !TFHE.glwe<{750,1,64}{4}>
//CHECK:   %[[V2:.*]] = "TFHE.bootstrap_glwe"(%[[V1]], %[[V0]]) {baseLog = 23 : i32, level = 1 : i32} : (!TFHE.glwe<{750,1,64}{4}>, !TFHE.glwe<{2,1024,64}{4}>) -> !TFHE.glwe<{2048,1,64}{4}>
//CHECK:   return %[[V2]] : !TFHE.glwe<{2048,1,64}{4}>
//CHECK: }
func @main(%arg0: !TFHE.glwe<{_,_,_}{4}>) -> !TFHE.glwe<{_,_,_}{4}> {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
  %0 = "TFHE.glwe_from_table"(%cst) : (tensor<16xi64>) -> !TFHE.glwe<{_,_,_}{4}>
  %1 = "TFHE.keyswitch_glwe"(%arg0) {baseLog = -1 : i32, level = -1 : i32} : (!TFHE.glwe<{_,_,_}{4}>) -> !TFHE.glwe<{_,_,_}{4}>
  %2 = "TFHE.bootstrap_glwe"(%1, %0) {baseLog = -1 : i32, level = -1 : i32} : (!TFHE.glwe<{_,_,_}{4}>, !TFHE.glwe<{_,_,_}{4}>) -> !TFHE.glwe<{_,_,_}{4}>
  return %2 : !TFHE.glwe<{_,_,_}{4}>
}
