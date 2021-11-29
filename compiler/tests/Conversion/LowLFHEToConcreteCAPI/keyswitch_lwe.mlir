// RUN: zamacompiler --passes lowlfhe-to-concrete-c-api --action=dump-std %s 2>&1| FileCheck %s

// CHECK-LABEL: module
// CHECK: func private @keyswitch_lwe_u64(index, !LowLFHE.lwe_key_switch_key, !LowLFHE.lwe_ciphertext<_,_>, !LowLFHE.lwe_ciphertext<_,_>)
// CHECK: func private @get_keyswitch_key(!LowLFHE.context) -> !LowLFHE.lwe_key_switch_key
// CHECK: func private @bootstrap_lwe_u64(index, !LowLFHE.lwe_bootstrap_key, !LowLFHE.lwe_ciphertext<_,_>, !LowLFHE.lwe_ciphertext<_,_>, !LowLFHE.glwe_ciphertext)
// CHECK: func private @get_bootstrap_key(!LowLFHE.context) -> !LowLFHE.lwe_bootstrap_key
// CHECK: func private @allocate_lwe_ciphertext_u64(index, index) -> !LowLFHE.lwe_ciphertext<_,_>
// CHECK-LABEL: func @keyswitch_lwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>, %arg1: !LowLFHE.context) -> !LowLFHE.lwe_ciphertext<1024,4>
func @keyswitch_lwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<1024,4> {
  // CHECK-NEXT: %[[ERR:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 1025 : index
  // CHECK-NEXT: %[[V1:.*]] = call @allocate_lwe_ciphertext_u64(%[[ERR]], %[[C0]]) : (index, index) -> !LowLFHE.lwe_ciphertext<_,_>
  // CHECK-NEXT: %[[V2:.*]] = call @get_keyswitch_key(%arg1) : (!LowLFHE.context) -> !LowLFHE.lwe_key_switch_key
  // CHECK-NEXT: %[[V3:.*]] = builtin.unrealized_conversion_cast %arg0 : !LowLFHE.lwe_ciphertext<1024,4> to !LowLFHE.lwe_ciphertext<_,_>
  // CHECK-NEXT: call @keyswitch_lwe_u64(%[[ERR]], %[[V2]], %[[V1]], %[[V3]]) : (index, !LowLFHE.lwe_key_switch_key, !LowLFHE.lwe_ciphertext<_,_>, !LowLFHE.lwe_ciphertext<_,_>) -> ()
  // CHECK-NEXT: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[V1]] : !LowLFHE.lwe_ciphertext<_,_> to !LowLFHE.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[RES]] : !LowLFHE.lwe_ciphertext<1024,4>
  %1 = "LowLFHE.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, inputLweSize = 1 : i32, level = 3 : i32, outputLweSize = 1 : i32} : (!LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<1024,4>
  return %1: !LowLFHE.lwe_ciphertext<1024,4>
}
