// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK: func.func @keyswitch_glwe(%[[A0:.*]]: !TFHE.glwe<sk[1]<1024,1>>) -> !TFHE.glwe<sk[1]<527,1>> {
func.func @keyswitch_glwe(%arg0: !TFHE.glwe<sk[1]<1024,1>>) -> !TFHE.glwe<sk[1]<527,1>> {
  // CHECK-NEXT: %[[V0:.*]] = "TFHE.keyswitch_glwe"(%[[A0]]) {key = #TFHE.ksk<sk[1]<1024,1>, sk[1]<527,1>, 4, 4>} : (!TFHE.glwe<sk[1]<1024,1>>) -> !TFHE.glwe<sk[1]<527,1>>
  // CHECK-NEXT: return %[[V0]] : !TFHE.glwe<sk[1]<527,1>>
  %0 = "TFHE.keyswitch_glwe"(%arg0) {key=#TFHE.ksk<sk[1]<1024,1>,sk[1]<527,1>,4,4>} : (!TFHE.glwe<sk[1]<1024,1>>) -> !TFHE.glwe<sk[1]<527,1>>
  return %0: !TFHE.glwe<sk[1]<527,1>>
}

// CHECK: func.func @bootstrap_glwe(%[[GLWE:.*]]: !TFHE.glwe<sk[1]<527,1>>, %[[LUT:.*]]: tensor<512xi64>) -> !TFHE.glwe<sk[1]<1024,1>> {
func.func @bootstrap_glwe(%glwe: !TFHE.glwe<sk[1]<527,1>>, %lut: tensor<512xi64>) -> !TFHE.glwe<sk[1]<1024,1>> {
    // CHECK-NEXT: %[[V0:.*]] = "TFHE.bootstrap_glwe"(%[[GLWE]], %[[LUT]]) {key = #TFHE.bsk<sk[1]<527,1>, sk[1]<1024,1>, 512, 2, 4, 4>} : (!TFHE.glwe<sk[1]<527,1>>, tensor<512xi64>) -> !TFHE.glwe<sk[1]<1024,1>>
    // CHECK-NEXT: return %[[V0]] : !TFHE.glwe<sk[1]<1024,1>>
    %0 = "TFHE.bootstrap_glwe"(%glwe, %lut) {key=#TFHE.bsk<sk[1]<527,1>,sk[1]<1024,1>,512,2,4,4>} : (!TFHE.glwe<sk[1]<527,1>>, tensor<512xi64>) -> !TFHE.glwe<sk[1]<1024,1>>
    return %0 : !TFHE.glwe<sk[1]<1024,1>>
}

