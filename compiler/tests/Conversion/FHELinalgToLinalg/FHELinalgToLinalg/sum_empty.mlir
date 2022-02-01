// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK:      func @sum_empty(%[[a0:.*]]: tensor<0x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero"() : () -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func @sum_empty(%arg0: tensor<0x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<0x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}
