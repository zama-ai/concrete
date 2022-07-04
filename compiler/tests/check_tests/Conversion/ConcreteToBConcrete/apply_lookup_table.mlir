// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

//CHECK: func.func @apply_lookup_table(%[[A0:.*]]: tensor<1025xi64>, %[[A1:.*]]: tensor<16xi64>) -> tensor<1025xi64> {
//CHECK:   %[[V0:.*]] = bufferization.alloc_tensor() : tensor<2048xi64>
//CHECK:   "BConcrete.fill_glwe_from_table"(%[[V0]], %[[A1]]) {glweDimension = 1 : i32, outPrecision = 4 : i32, polynomialSize = 1024 : i32} : (tensor<2048xi64>, tensor<16xi64>) -> ()
//CHECK:   %[[V1:.*]] = "BConcrete.keyswitch_lwe_buffer"(%[[A0]]) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 600 : i32} : (tensor<1025xi64>) -> tensor<601xi64>
//CHECK:   %[[V2:.*]] = "BConcrete.bootstrap_lwe_buffer"(%[[V1]], %[[V0]]) {baseLog = 4 : i32, glweDimension = 1 : i32, level = 5 : i32, polynomialSize = 1024 : i32} : (tensor<601xi64>, tensor<2048xi64>) -> tensor<1025xi64>
//CHECK:   return %[[V2]] : tensor<1025xi64>
//CHECK: }
func.func @apply_lookup_table(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: tensor<16xi64>) -> !Concrete.lwe_ciphertext<1024,4> {
  %0 = "Concrete.glwe_from_table"(%arg1) {glweDimension = 1 : i32, p = 4 : i32, polynomialSize = 1024 : i32} : (tensor<16xi64>) -> !Concrete.glwe_ciphertext<1,1024,4>
  %1 = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 600 : i32} : (!Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<600,4>
  %2 = "Concrete.bootstrap_lwe"(%1, %0) {baseLog = 4 : i32, glweDimension = 1 : i32, level = 5 : i32, polynomialSize = 1024 : i32} : (!Concrete.lwe_ciphertext<600,4>, !Concrete.glwe_ciphertext<1,1024,4>) -> !Concrete.lwe_ciphertext<1024,4>
  return %2 : !Concrete.lwe_ciphertext<1024,4>
}
