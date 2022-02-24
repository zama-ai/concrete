// RUN: concretecompiler --passes bconcrete-to-bconcrete-c-api --action=dump-std %s 2>&1| FileCheck %s

// CHECK: func @apply_lookup_table(%arg0: tensor<601xi64>, %arg1: tensor<2048xi64>, %arg2: !Concrete.context) -> tensor<1025xi64> {
// CHECK-NEXT: %0 = linalg.init_tensor [1025] : tensor<1025xi64>
// CHECK-NEXT: %1 = tensor.cast %0 : tensor<1025xi64> to tensor<?xi64>
// CHECK-NEXT: %2 = tensor.cast %arg0 : tensor<601xi64> to tensor<?xi64>
// CHECK-NEXT: %3 = tensor.cast %arg1 : tensor<2048xi64> to tensor<?xi64>
// CHECK-NEXT: call @memref_bootstrap_lwe_u64(%1, %2, %3, %arg2) : (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, !Concrete.context) -> ()
// CHECK-NEXT: return %0 : tensor<1025xi64>
// CHECK-NEXT: }
func @apply_lookup_table(%arg0: tensor<601xi64>, %arg1: tensor<2048xi64>) -> tensor<1025xi64> {
    %0 = linalg.init_tensor [1025] : tensor<1025xi64>
    "BConcrete.bootstrap_lwe_buffer"(%0, %arg0, %arg1) {baseLog = 4 : i32, glweDimension = 1 : i32, level = 5 : i32, polynomialSize = 1024 : i32} : (tensor<1025xi64>, tensor<601xi64>, tensor<2048xi64>) -> ()
    return %0 : tensor<1025xi64>
  }