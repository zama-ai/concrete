// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func @apply_lookup_table(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: tensor<16xi64>) -> !Concrete.lwe_ciphertext<1024,4>
func @apply_lookup_table(%arg0: !TFHE.glwe<{1024,1,64}{4}>, %arg1: tensor<16xi64>) -> !TFHE.glwe<{1024,1,64}{4}> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.glwe_from_table"(%arg1) : (tensor<16xi64>) -> !Concrete.glwe_ciphertext<1024,1,4>
  // CHECK-NEXT: %[[V2:.*]] = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, level = 3 : i32} : (!Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<600,4>
  // CHECK-NEXT: %[[V3:.*]] = "Concrete.bootstrap_lwe"(%[[V2]], %[[V1]]) {baseLog = 4 : i32, glweDimension = 1 : i32, level = 5 : i32, polynomialSize = 1024 : i32} : (!Concrete.lwe_ciphertext<600,4>, !Concrete.glwe_ciphertext<1024,1,4>) -> !Concrete.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[V3]] : !Concrete.lwe_ciphertext<1024,4>
  %1 = "TFHE.apply_lookup_table"(%arg0, %arg1){glweDimension=1:i32, polynomialSize=1024:i32, levelKS=3:i32, baseLogKS=2:i32, levelBS=5:i32, baseLogBS=4:i32, outputSizeKS=600:i32}: (!TFHE.glwe<{1024,1,64}{4}>, tensor<16xi64>) -> (!TFHE.glwe<{1024,1,64}{4}>)
  return %1: !TFHE.glwe<{1024,1,64}{4}>
}
