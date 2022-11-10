// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK:  func.func @apply_lookup_table(%arg0: tensor<4xi64>) -> tensor<1024xi64> {
// CHECK-NEXT:    %0 = "Concrete.encode_expand_lut_for_bootstrap"(%arg0) {outputBits = 3 : i32, polySize = 1024 : i32} : (tensor<4xi64>) -> tensor<1024xi64>
// CHECK-NEXT:    return %0 : tensor<1024xi64>
// CHECK-NEXT:  }
func.func @apply_lookup_table(%arg1: tensor<4xi64>) -> tensor<1024xi64> {
    %0 = "TFHE.encode_expand_lut_for_bootstrap"(%arg1) {outputBits = 3 : i32, polySize = 1024 : i32} : (tensor<4xi64>) -> tensor<1024xi64>
    return %0: tensor<1024xi64>
}
