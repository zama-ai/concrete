// RUN: concretecompiler --action=dump-llvm-dialect --emit-gpu-ops --skip-program-info %s 2>&1| FileCheck %s

//CHECK: llvm.call @memref_keyswitch_lwe_cuda_u64
//CHECK: llvm.call @memref_bootstrap_lwe_cuda_u64
func.func @main(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %0 = "Concrete.keyswitch_lwe_tensor"(%arg0) {baseLog = 2 : i32, kskIndex = 0 : i32, level = 5 : i32, lwe_dim_in = 1025 : i32, lwe_dim_out = 576 : i32} : (tensor<1025xi64>) -> tensor<576xi64>
  %1 = "Concrete.bootstrap_lwe_tensor"(%0, %cst) {baseLog = 2 : i32, bskIndex = 0 : i32, level = 5 : i32, polySize = 1024: i32, glweDimension = 1 : i32, inputLweDim = 576 : i32, outPrecision = 2 : i32} : (tensor<576xi64>, tensor<4xi64>) -> tensor<1025xi64>
  return %1 : tensor<1025xi64>
}
