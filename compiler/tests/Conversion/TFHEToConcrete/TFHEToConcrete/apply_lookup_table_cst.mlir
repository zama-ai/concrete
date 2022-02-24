// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func @apply_lookup_table_cst(%arg0: !Concrete.lwe_ciphertext<2048,4>) -> !Concrete.lwe_ciphertext<2048,4>
func @apply_lookup_table_cst(%arg0: !TFHE.glwe<{2048,1,64}{4}>) -> !TFHE.glwe<{2048,1,64}{4}> {
  // CHECK-NEXT: %[[TABLE:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.glwe_from_table"(%[[TABLE]]) : (tensor<16xi64>) -> !Concrete.glwe_ciphertext<2048,1,4>
  // CHECK-NEXT: %[[V2:.*]] = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, level = 3 : i32} : (!Concrete.lwe_ciphertext<2048,4>) -> !Concrete.lwe_ciphertext<600,4>
  // CHECK-NEXT: %[[V3:.*]] = "Concrete.bootstrap_lwe"(%[[V2]], %[[V1]]) {baseLog = 4 : i32, glweDimension = 1 : i32, level = 5 : i32, polynomialSize = 2048 : i32} : (!Concrete.lwe_ciphertext<600,4>, !Concrete.glwe_ciphertext<2048,1,4>) -> !Concrete.lwe_ciphertext<2048,4>
  // CHECK-NEXT: return %[[V3]] : !Concrete.lwe_ciphertext<2048,4>
  %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
  %1 = "TFHE.apply_lookup_table"(%arg0, %tlu){glweDimension=1:i32, polynomialSize=2048:i32, levelKS=3:i32, baseLogKS=2:i32, levelBS=5:i32, baseLogBS=4:i32, outputSizeKS=600:i32}: (!TFHE.glwe<{2048,1,64}{4}>, tensor<16xi64>) -> (!TFHE.glwe<{2048,1,64}{4}>)
  return %1: !TFHE.glwe<{2048,1,64}{4}>
}
