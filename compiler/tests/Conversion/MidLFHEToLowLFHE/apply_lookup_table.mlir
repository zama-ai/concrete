// RUN: zamacompiler --entry-dialect=midlfhe --action=dump-lowlfhe --parametrize-midlfhe=false --assume-max-eint-precision=7 --assume-max-manp=10 %s 2>&1| FileCheck %s

// CHECK-LABEL: func @apply_lookup_table(%arg0: !LowLFHE.lwe_ciphertext<1024,4>, %arg1: tensor<16xi64>) -> !LowLFHE.lwe_ciphertext<1024,4>
func @apply_lookup_table(%arg0: !MidLFHE.glwe<{1024,1,64}{4}>, %arg1: tensor<16xi64>) -> !MidLFHE.glwe<{1024,1,64}{4}> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.glwe_from_table"(%arg1) {k = 1 : i32, p = 4 : i32, polynomialSize = 1024 : i32} : (tensor<16xi64>) -> !LowLFHE.glwe_ciphertext
  // CHECK-NEXT: %[[V2:.*]] = "LowLFHE.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, inputLweSize = 1 : i32, level = 3 : i32, outputLweSize = 600 : i32} : (!LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<600,4>
  // CHECK-NEXT: %[[V3:.*]] = "LowLFHE.bootstrap_lwe"(%[[V2]], %[[V1]]) {baseLog = 4 : i32, k = 1 : i32, level = 5 : i32, polynomialSize = 1024 : i32} : (!LowLFHE.lwe_ciphertext<600,4>, !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[V3]] : !LowLFHE.lwe_ciphertext<1024,4>
  %1 = "MidLFHE.apply_lookup_table"(%arg0, %arg1){k=1:i32, polynomialSize=1024:i32, levelKS=3:i32, baseLogKS=2:i32, levelBS=5:i32, baseLogBS=4:i32, outputSizeKS=600:i32}: (!MidLFHE.glwe<{1024,1,64}{4}>, tensor<16xi64>) -> (!MidLFHE.glwe<{1024,1,64}{4}>)
  return %1: !MidLFHE.glwe<{1024,1,64}{4}>
}
