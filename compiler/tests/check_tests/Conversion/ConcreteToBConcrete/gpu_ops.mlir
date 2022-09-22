// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete --emit-gpu-ops %s 2>&1| FileCheck %s


//CHECK: func.func @main(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
//CHECK:   %c1_i32 = arith.constant 1 : i32
//CHECK:   %c8_i32 = arith.constant 8 : i32
//CHECK:   %c2_i32 = arith.constant 2 : i32
//CHECK:   %c1024_i32 = arith.constant 1024 : i32
//CHECK:   %c575_i32 = arith.constant 575 : i32
//CHECK:   %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
//CHECK:   %c5_i32 = arith.constant 5 : i32
//CHECK:   %c2_i32_0 = arith.constant 2 : i32
//CHECK:   %c575_i32_1 = arith.constant 575 : i32
//CHECK:   %c1024_i32_2 = arith.constant 1024 : i32
//CHECK:   %0 = "BConcrete.keyswitch_lwe_gpu_buffer"(%arg0, %c5_i32, %c2_i32_0, %c1024_i32_2, %c575_i32_1) : (tensor<1025xi64>, i32, i32, i32, i32) -> tensor<576xi64>
//CHECK:   %1 = "BConcrete.bootstrap_lwe_gpu_buffer"(%0, %cst, %c575_i32, %c1024_i32, %c2_i32, %c8_i32, %c1_i32, %c2_i32) : (tensor<576xi64>, tensor<4xi64>, i32, i32, i32, i32, i32, i32) -> tensor<1025xi64>
//CHECK:   return %1 : tensor<1025xi64>
//CHECK: }
func.func @main(%arg0: !Concrete.lwe_ciphertext<1024,2>) -> !Concrete.lwe_ciphertext<1024,2> {
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c575_i32 = arith.constant 575 : i32
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %0 = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, level = 5 : i32} : (!Concrete.lwe_ciphertext<1024,2>) -> !Concrete.lwe_ciphertext<575,2>
  %1 = "Concrete.bootstrap_lwe"(%0, %cst, %c575_i32, %c1024_i32, %c2_i32, %c8_i32, %c1_i32, %c2_i32) : (!Concrete.lwe_ciphertext<575,2>, tensor<4xi64>, i32, i32, i32, i32, i32, i32) -> !Concrete.lwe_ciphertext<1024,2>
  return %1 : !Concrete.lwe_ciphertext<1024,2>
}


