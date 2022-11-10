// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

// CHECK:  func.func @main(%arg0: i64) -> tensor<5xi64> {
// CHECK-NEXT:    %0 = "BConcrete.encode_plaintext_with_crt_tensor"(%arg0) {mods = [2, 3, 5, 7, 11], modsProd = 2310 : i64} : (i64) -> tensor<5xi64>
// CHECK-NEXT:    return %0 : tensor<5xi64>
// CHECK-NEXT:  }
func.func @main(%arg0: i64) -> tensor<5xi64> {
  %0 = "Concrete.encode_plaintext_with_crt"(%arg0) {mods = [2, 3, 5, 7, 11], modsProd = 2310 : i64} : (i64) -> tensor<5xi64>
  return %0 : tensor<5xi64>
}
