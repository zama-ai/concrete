// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK: func.func @keyswitch_glwe(%[[A0:.*]]: tensor<1025xi64>) -> tensor<568xi64> {
// CHECK-NEXT:   %[[V0:.*]] = "Concrete.keyswitch_lwe_tensor"(%[[A0]]) {baseLog = 3 : i32, level = 2 : i32, lwe_dim_in = 1024 : i32, lwe_dim_out = 567 : i32} : (tensor<1025xi64>) -> tensor<568xi64>
// CHECK-NEXT:   return %[[V0]] : tensor<568xi64>
// CHECK-NEXT: }
func.func @keyswitch_glwe(%arg0: !TFHE.glwe<{1024,1,64}{2}>) -> !TFHE.glwe<{567,1,64}{2}> {
  %0 = "TFHE.keyswitch_glwe"(%arg0) {baseLog = 3 : i32, level = 2 : i32} : (!TFHE.glwe<{1024,1,64}{2}>) -> !TFHE.glwe<{567,1,64}{2}>
  return %0 : !TFHE.glwe<{567,1,64}{2}>
}
