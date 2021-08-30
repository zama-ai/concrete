// RUN: zamacompiler --passes lowlfhe-to-concrete-c-api %s  2>&1| FileCheck %s

// CHECK-LABEL: module
// CHECK-NEXT: func private @keyswitch_lwe_u64(index, !LowLFHE.lwe_key_switch_key, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.lwe_ciphertext<1024,4>)
// CHECK-NEXT: func private @allocate_lwe_ciphertext_u64(index, i32) -> !LowLFHE.lwe_ciphertext<1024,4>
// CHECK-LABEL: func @keyswitch_lwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<1024,4>
func @keyswitch_lwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<1024,4> {
  // CHECK-NEXT: %[[ERR:.*]] = constant 0 : index
  // CHECK-NEXT: %[[C0:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V1:.*]] = call @allocate_lwe_ciphertext_u64(%[[ERR]], %[[C0]]) : (index, i32) -> !LowLFHE.lwe_ciphertext<1024,4>
  // CHECK-NEXT: %[[V2:.*]] = call @getGlobalKeyswitchKey() : () -> !LowLFHE.lwe_key_switch_key
  // CHECK-NEXT: call @keyswitch_lwe_u64(%[[ERR]], %[[V2]], %[[V1]], %arg0) : (index, !LowLFHE.lwe_key_switch_key, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.lwe_ciphertext<1024,4>) -> ()
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<1024,4>
  %1 = "LowLFHE.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, inputLweSize = 1 : i32, level = 3 : i32, outputLweSize = 1 : i32} : (!LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<1024,4>
  return %1: !LowLFHE.lwe_ciphertext<1024,4>
}