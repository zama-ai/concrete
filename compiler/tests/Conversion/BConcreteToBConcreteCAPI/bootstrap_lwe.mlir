// RUN: concretecompiler --passes bconcrete-to-bconcrete-c-api --action=dump-std %s 2>&1| FileCheck %s

// CHECK: func @bootstrap_lwe(%arg0: tensor<1025xi64>, %arg1: !Concrete.glwe_ciphertext, %arg2: !Concrete.context) -> tensor<1025xi64> {
// CHECK-NEXT:   %0 = linalg.init_tensor [1025] : tensor<1025xi64>
// CHECK-NEXT:   %1 = call @get_bootstrap_key(%arg2) : (!Concrete.context) -> !Concrete.lwe_bootstrap_key
// CHECK-NEXT:   %2 = tensor.cast %0 : tensor<1025xi64> to tensor<?xi64>
// CHECK-NEXT:   %3 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
// CHECK-NEXT:   call @memref_bootstrap_lwe_u64(%1, %2, %3, %arg1) : (!Concrete.lwe_bootstrap_key, tensor<?xi64>, tensor<?xi64>, !Concrete.glwe_ciphertext) -> ()
// CHECK-NEXT:   return %0 : tensor<1025xi64>
// CHECK-NEXT: }
func @bootstrap_lwe(%arg0: tensor<1025xi64>, %arg1: !Concrete.glwe_ciphertext) -> tensor<1025xi64> {
  %0 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.bootstrap_lwe_buffer"(%0, %arg0, %arg1) {baseLog = 2 : i32, glweDimension = 1 : i32, level = 3 : i32, polynomialSize = 1024 : i32} : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.glwe_ciphertext) -> ()
  return %0 : tensor<1025xi64>
}