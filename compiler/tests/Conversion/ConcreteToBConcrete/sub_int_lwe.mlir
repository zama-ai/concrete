// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func @sub_const_int_lwe(%arg0: tensor<1025xi64>) -> tensor<1025xi64>
func @sub_const_int_lwe(%arg0: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: "BConcrete.negate_lwe_buffer"(%[[V2]], %arg0) : (tensor<1025xi64>, tensor<1025xi64>) -> ()
  // CHECK-NEXT: %[[V3:.*]] = "Concrete.encode_int"(%[[V1]]) : (i8) -> !Concrete.plaintext<8>
  // CHECK-NEXT: %[[V4:.*]] = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: "BConcrete.add_plaintext_lwe_buffer"(%[[V4]], %[[V2]], %[[V3]]) : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.plaintext<8>) -> ()
  // CHECK-NEXT: return %[[V4]] : tensor<1025xi64>
  %0 = arith.constant 1 : i8
  %1 = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7>
  %2 = "Concrete.encode_int"(%0) : (i8) -> !Concrete.plaintext<8>
  %3 = "Concrete.add_plaintext_lwe_ciphertext"(%1, %2) : (!Concrete.lwe_ciphertext<1024,7>, !Concrete.plaintext<8>) -> !Concrete.lwe_ciphertext<1024,7>
  return %3 : !Concrete.lwe_ciphertext<1024,7>
}


// CHECK-LABEL: func @sub_int_lwe(%arg0: tensor<1025xi64>, %arg1: i5) -> tensor<1025xi64>
func @sub_int_lwe(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: i5) -> !Concrete.lwe_ciphertext<1024,4> {
  // CHECK-NEXT: %[[V1:.*]] = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: "BConcrete.negate_lwe_buffer"(%[[V1]], %arg0) : (tensor<1025xi64>, tensor<1025xi64>) -> ()
  // CHECK-NEXT: %[[V2:.*]] = "Concrete.encode_int"(%arg1) : (i5) -> !Concrete.plaintext<5>
  // CHECK-NEXT: %[[V3:.*]] = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: "BConcrete.add_plaintext_lwe_buffer"(%[[V3]], %[[V1]], %[[V2]]) : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.plaintext<5>) -> ()
  // CHECK-NEXT: return %[[V3]] : tensor<1025xi64>
  %0 = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4>
  %1 = "Concrete.encode_int"(%arg1) : (i5) -> !Concrete.plaintext<5>
  %2 = "Concrete.add_plaintext_lwe_ciphertext"(%0, %1) : (!Concrete.lwe_ciphertext<1024,4>, !Concrete.plaintext<5>) -> !Concrete.lwe_ciphertext<1024,4>
  return %2 : !Concrete.lwe_ciphertext<1024,4>
}
