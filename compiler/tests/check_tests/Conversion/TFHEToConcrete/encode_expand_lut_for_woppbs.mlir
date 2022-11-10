// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK:  func.func @main(%arg0: tensor<4xi64>) -> tensor<40960xi64> {
// CHECK-NEXT:    %0 = "Concrete.encode_expand_lut_for_woppbs"(%arg0) {crtBits = [1, 2, 3, 3, 4], crtDecomposition = [2, 3, 5, 7, 11], modulusProduct = 2310 : i32, polySize = 1024 : i32} : (tensor<4xi64>) -> tensor<40960xi64>
// CHECK-NEXT:    return %0 : tensor<40960xi64>
// CHECK-NEXT:  }
func.func @main(%arg1: tensor<4xi64>) -> tensor<40960xi64> {
    %0 = "TFHE.encode_expand_lut_for_woppbs"(%arg1) {crtBits = [1, 2, 3, 3, 4], crtDecomposition = [2, 3, 5, 7, 11], modulusProduct = 2310 : i32, polySize = 1024 : i32} : (tensor<4xi64>) -> tensor<40960xi64>
    return %0: tensor<40960xi64>
}
