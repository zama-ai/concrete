// RUN: concretecompiler --passes bconcrete-to-bconcrete-c-api --action=dump-std %s 2>&1| FileCheck %s

//CHECK: func @keyswitch_lwe(%arg0: tensor<1025xi64>, %arg1: !Concrete.context) -> tensor<1025xi64> {
//CHECK-NEXT:   %0 = linalg.init_tensor [1025] : tensor<1025xi64>
//CHECK-NEXT:   %1 = tensor.cast %0 : tensor<1025xi64> to tensor<?xi64>
//CHECK-NEXT:   %2 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
//CHECK-NEXT:   call @memref_keyswitch_lwe_u64(%1, %2, %arg1) : (tensor<?xi64>, tensor<?xi64>, !Concrete.context) -> ()
//CHECK-NEXT:   return %0 : tensor<1025xi64>
//CHECK-NEXT: }
func @keyswitch_lwe(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
  %0 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.keyswitch_lwe_buffer"(%0, %arg0) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 1 : i32} : (tensor<1025xi64>, tensor<1025xi64>) -> ()
  return %0 : tensor<1025xi64>
}