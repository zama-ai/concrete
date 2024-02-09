// RUN: concretecompiler --action=dump-parametrized-tfhe --optimizer-strategy=V0 --v0-parameter=2,10,750,1,23,3,4 --v0-constraint=4,0 --skip-program-info %s 2>&1| FileCheck %s

//CHECK: func.func @main(%[[A0:.*]]: !TFHE.glwe<sk<0,1,2048>>) -> !TFHE.glwe<sk<0,1,2048>> {
//CHECK-NEXT:   %cst = arith.constant dense<0> : tensor<1024xi64>
//CHECK-NEXT:   %[[V1:.*]] = "TFHE.keyswitch_glwe"(%[[A0]]) {key = #TFHE.ksk<sk<0,1,2048>, sk<1,1,750>, 3, 4>} : (!TFHE.glwe<sk<0,1,2048>>) -> !TFHE.glwe<sk<1,1,750>>
//CHECK-NEXT:   %[[V2:.*]] = "TFHE.bootstrap_glwe"(%[[V1]], %cst) {key = #TFHE.bsk<sk<1,1,750>, sk<0,1,2048>, 1024, 2, 1, 23>} : (!TFHE.glwe<sk<1,1,750>>, tensor<1024xi64>) -> !TFHE.glwe<sk<0,1,2048>>
//CHECK-NEXT:   return %[[V2]] : !TFHE.glwe<sk<0,1,2048>>
//CHECK-NEXT: }
func.func @main(%arg0: !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?> {
  %cst = arith.constant dense<0> : tensor<1024xi64>
  %1 = "TFHE.keyswitch_glwe"(%arg0) {key = #TFHE.ksk<sk?, sk?, -1, -1>} : (!TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
  %2 = "TFHE.bootstrap_glwe"(%1, %cst) {key = #TFHE.bsk<sk?, sk?, -1, -1, -1, -1>} : (!TFHE.glwe<sk?>, tensor<1024xi64>) -> !TFHE.glwe<sk?>
  return %2 : !TFHE.glwe<sk?>
}
