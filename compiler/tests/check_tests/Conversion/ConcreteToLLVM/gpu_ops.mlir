// RUN: concretecompiler --action=dump-llvm-dialect --emit-gpu-ops %s 2>&1| FileCheck %s

//CHECK: llvm.call @memref_keyswitch_lwe_cuda_u64
//CHECK: llvm.call @memref_bootstrap_lwe_cuda_u64
func.func @main(%arg0: !Concrete.lwe_ciphertext<1024,2>) -> !Concrete.lwe_ciphertext<1024,2> {
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %0 = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, level = 5 : i32} : (!Concrete.lwe_ciphertext<1024,2>) -> !Concrete.lwe_ciphertext<575,2>
  %1 = "Concrete.bootstrap_lwe"(%0, %cst) {baseLog = 2 : i32, level = 5 : i32, polySize = 1024: i32, glweDimension = 1 : i32} : (!Concrete.lwe_ciphertext<575,2>, tensor<4xi64>) -> !Concrete.lwe_ciphertext<1024,2>
  return %1 : !Concrete.lwe_ciphertext<1024,2>
}
