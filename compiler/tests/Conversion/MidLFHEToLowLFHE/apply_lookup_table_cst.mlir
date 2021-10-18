// RUN: zamacompiler --passes midlfhe-to-lowlfhe --action=dump-lowlfhe %s 2>&1| FileCheck %s

// CHECK-LABEL: func @apply_lookup_table_cst(%arg0: !LowLFHE.lwe_ciphertext<2048,4>) -> !LowLFHE.lwe_ciphertext<2048,4>
func @apply_lookup_table_cst(%arg0: !MidLFHE.glwe<{2048,1,64}{4}>) -> !MidLFHE.glwe<{2048,1,64}{4}> {
  // CHECK-NEXT: %[[TABLE:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.glwe_from_table"(%[[TABLE]]) {k = 1 : i32, p = 4 : i32, polynomialSize = 2048 : i32} : (tensor<16xi64>) -> !LowLFHE.glwe_ciphertext
  // CHECK-NEXT: %[[V2:.*]] = "LowLFHE.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, inputLweSize = 1 : i32, level = 3 : i32, outputLweSize = 600 : i32} : (!LowLFHE.lwe_ciphertext<2048,4>) -> !LowLFHE.lwe_ciphertext<600,4>
  // CHECK-NEXT: %[[V3:.*]] = "LowLFHE.bootstrap_lwe"(%[[V2]], %[[V1]]) {baseLog = 4 : i32, k = 1 : i32, level = 5 : i32, polynomialSize = 2048 : i32} : (!LowLFHE.lwe_ciphertext<600,4>, !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<2048,4>
  // CHECK-NEXT: return %[[V3]] : !LowLFHE.lwe_ciphertext<2048,4>
  %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
  %1 = "MidLFHE.apply_lookup_table"(%arg0, %tlu){k=1:i32, polynomialSize=2048:i32, levelKS=3:i32, baseLogKS=2:i32, levelBS=5:i32, baseLogBS=4:i32, outputSizeKS=600:i32}: (!MidLFHE.glwe<{2048,1,64}{4}>, tensor<16xi64>) -> (!MidLFHE.glwe<{2048,1,64}{4}>)
  return %1: !MidLFHE.glwe<{2048,1,64}{4}>
}
