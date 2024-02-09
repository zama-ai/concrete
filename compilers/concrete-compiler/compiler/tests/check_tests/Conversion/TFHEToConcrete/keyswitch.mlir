// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete --skip-program-info %s 2>&1| FileCheck %s

// CHECK: func.func @keyswitch_glwe(%[[A0:.*]]: tensor<1025xi64>) -> tensor<568xi64> {
// CHECK-NEXT:   %[[V0:.*]] = "Concrete.keyswitch_lwe_tensor"(%[[A0]]) {baseLog = 3 : i32, kskIndex = -1 : i32, level = 2 : i32, lwe_dim_in = 1024 : i32, lwe_dim_out = 567 : i32} : (tensor<1025xi64>) -> tensor<568xi64>
// CHECK-NEXT:   return %[[V0]] : tensor<568xi64>
// CHECK-NEXT: }
func.func @keyswitch_glwe(%arg0: !TFHE.glwe<sk[1]<1,1024>>) -> !TFHE.glwe<sk[3]<1,567>> {
  %0 = "TFHE.keyswitch_glwe"(%arg0) {key = #TFHE.ksk<sk[1]<1,1024>, sk[3]<1,567>, 2, 3>} : (!TFHE.glwe<sk[1]<1,1024>>) -> !TFHE.glwe<sk[3]<1,567>>
  return %0 : !TFHE.glwe<sk[3]<1,567>>
}
