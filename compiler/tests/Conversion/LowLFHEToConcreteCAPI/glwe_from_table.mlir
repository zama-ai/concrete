// RUN: zamacompiler --passes lowlfhe-to-concrete-c-api --action=dump-std %s 2>&1| FileCheck %s

// CHECK-LABEL: module
// CHECK: func private @runtime_foreign_plaintext_list_u64(index, tensor<16xi64>, i64, i32) -> !LowLFHE.foreign_plaintext_list
// CHECK: func private @add_plaintext_list_glwe_ciphertext_u64(index, !LowLFHE.glwe_ciphertext, !LowLFHE.glwe_ciphertext, !LowLFHE.plaintext_list)
// CHECK: func private @fill_plaintext_list_with_expansion_u64(index, !LowLFHE.plaintext_list, !LowLFHE.foreign_plaintext_list)
// CHECK: func private @allocate_plaintext_list_u64(index, i32) -> !LowLFHE.plaintext_list
// CHECK: func private @allocate_glwe_ciphertext_u64(index, i32, i32) -> !LowLFHE.glwe_ciphertext
// CHECK-LABEL: func @glwe_from_table(%arg0: tensor<16xi64>, %arg1: !LowLFHE.context) -> !LowLFHE.glwe_ciphertext
func @glwe_from_table(%arg0: tensor<16xi64>) -> !LowLFHE.glwe_ciphertext {
  // CHECK-NEXT: %[[V0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 1 : i32
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1024 : i32
  // CHECK-NEXT: %[[V1:.*]] = call @allocate_glwe_ciphertext_u64(%[[V0]], %[[C0]], %[[C1]]) : (index, i32, i32) -> !LowLFHE.glwe_ciphertext
  // CHECK-NEXT: %[[V2:.*]] = call @allocate_glwe_ciphertext_u64(%[[V0]], %[[C0]], %[[C1]]) : (index, i32, i32) -> !LowLFHE.glwe_ciphertext
  // CHECK-NEXT: %[[V3:.*]] = call @allocate_plaintext_list_u64(%[[V0]], %[[C1]]) : (index, i32) -> !LowLFHE.plaintext_list
  // CHECK-NEXT: %[[C2:.*]] = arith.constant 16 : i64
  // CHECK-NEXT: %[[C3:.*]] = arith.constant 4 : i32
  // CHECK-NEXT: %[[V4:.*]] = call @runtime_foreign_plaintext_list_u64(%[[V0]], %arg0, %[[C2]], %[[C3]]) : (index, tensor<16xi64>, i64, i32) -> !LowLFHE.foreign_plaintext_list
  // CHECK-NEXT: call @fill_plaintext_list_with_expansion_u64(%[[V0]], %[[V3]], %[[V4]]) : (index, !LowLFHE.plaintext_list, !LowLFHE.foreign_plaintext_list) -> ()
  // CHECK-NEXT: call @add_plaintext_list_glwe_ciphertext_u64(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) : (index, !LowLFHE.glwe_ciphertext, !LowLFHE.glwe_ciphertext, !LowLFHE.plaintext_list) -> ()
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.glwe_ciphertext
  %1 = "LowLFHE.glwe_from_table"(%arg0) {k = 1 : i32, p = 4 : i32, polynomialSize = 1024 : i32} : (tensor<16xi64>) -> !LowLFHE.glwe_ciphertext
  return %1: !LowLFHE.glwe_ciphertext
}
