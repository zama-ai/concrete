// RUN: zamacompiler --passes lowlfhe-to-concrete-c-api %s  2>&1| FileCheck %s

// CHECK-LABEL: module
// CHECK-NEXT: func private @bootstrap_lwe_u64(memref<index>, !LowLFHE.lwe_bootstrap_key, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.glwe_ciphertext)
// CHECK-NEXT: func private @allocate_lwe_ciphertext_u64(memref<index>, i32) -> !LowLFHE.lwe_ciphertext<1024,4>
// CHECK-NEXT: func private @allocate_lwe_bootstrap_key_u64(memref<index>, i32, i32, i32, i32, i32) -> !LowLFHE.lwe_bootstrap_key
// CHECK-LABEL: func @bootstrap_lwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>, %arg1: !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<1024,4>
func @bootstrap_lwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>, %arg1: !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<1024,4> {
  // CHECK-NEXT: %[[V0:.*]] = memref.alloca() : memref<index>
  // CHECK-NEXT: %[[C0:.*]] = constant -1 : i32
  // CHECK-NEXT: %[[V1:.*]] = call @allocate_lwe_ciphertext_u64(%[[V0]], %[[C0]]) : (memref<index>, i32) -> !LowLFHE.lwe_ciphertext<1024,4>
  // CHECK-NEXT: %[[C1:.*]] = constant -1 : i32
  // CHECK-NEXT: %[[C2:.*]] = constant -1 : i32
  // CHECK-NEXT: %[[C3:.*]] = constant -1 : i32
  // CHECK-NEXT: %[[C4:.*]] = constant -1 : i32
  // CHECK-NEXT: %[[V2:.*]] = call @allocate_lwe_bootstrap_key_u64(%0, %[[C1]], %[[C2]], %[[C3]], %[[C0]], %[[C4]]) : (memref<index>, i32, i32, i32, i32, i32) -> !LowLFHE.lwe_bootstrap_key
  // CHECK-NEXT: call @bootstrap_lwe_u64(%[[V0]], %[[V2]], %[[V1]], %arg0, %arg1) : (memref<index>, !LowLFHE.lwe_bootstrap_key, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.glwe_ciphertext) -> ()
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<1024,4>
  %1 = "LowLFHE.bootstrap_lwe"(%arg0, %arg1) : (!LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<1024,4>
  return %1: !LowLFHE.lwe_ciphertext<1024,4>
}