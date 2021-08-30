// RUN: zamacompiler --passes lowlfhe-to-concrete-c-api %s  2>&1| FileCheck %s

// CHECK-LABEL: module
// CHECK-NEXT: func private @bootstrap_lwe_u64(index, !LowLFHE.lwe_bootstrap_key, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.glwe_ciphertext)
// CHECK-NEXT: func private @allocate_lwe_ciphertext_u64(index, i32) -> !LowLFHE.lwe_ciphertext<1024,4>
// CHECK-NEXT: func private @getGlobalBootstrapKey() -> !LowLFHE.lwe_bootstrap_key
// CHECK-LABEL: func @bootstrap_lwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>, %arg1: !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<1024,4>
func @bootstrap_lwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>, %arg1: !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<1024,4> {
  // CHECK-NEXT: %[[ERR:.*]] = constant 0 : index
  // CHECK-NEXT: %[[C0:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V1:.*]] = call @allocate_lwe_ciphertext_u64(%[[ERR]], %[[C0]]) : (index, i32) -> !LowLFHE.lwe_ciphertext<1024,4>
  // CHECK-NEXT: %[[V2:.*]] = call @getGlobalBootstrapKey() : () -> !LowLFHE.lwe_bootstrap_key
  // CHECK-NEXT: call @bootstrap_lwe_u64(%[[ERR]], %[[V2]], %[[V1]], %arg0, %arg1) : (index, !LowLFHE.lwe_bootstrap_key, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.glwe_ciphertext) -> ()
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<1024,4>
  %1 = "LowLFHE.bootstrap_lwe"(%arg0, %arg1) {baseLog = 2 : i32, k = 1 : i32, level = 3 : i32, polynomialSize = 1024 : i32} : (!LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<1024,4>
  return %1: !LowLFHE.lwe_ciphertext<1024,4>
}