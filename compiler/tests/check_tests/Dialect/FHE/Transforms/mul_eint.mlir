// RUN: concretecompiler --action=dump-fhe --split-input-file %s 2>&1 | FileCheck %s

// CHECK: func.func @simple_eint(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<3>) -> !FHE.eint<3> {
// CHECK-NEXT:     %cst = arith.constant dense<[0, 0, 1, 2, 4, 2, 1, 0]> : tensor<8xi64>
// CHECK-NEXT:     %cst_0 = arith.constant dense<[0, 0, 1, 2, 4, 6, 9, 12]> : tensor<8xi64>
// CHECK-NEXT:     %0 = "FHE.add_eint"(%arg0, %arg1) {MANP = 2 : ui3} : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
// CHECK-NEXT:     %1 = "FHE.apply_lookup_table"(%0, %cst_0) {MANP = 1 : ui1} : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<3>
// CHECK-NEXT:     %2 = "FHE.sub_eint"(%arg0, %arg1) {MANP = 2 : ui3} : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
// CHECK-NEXT:     %3 = "FHE.to_signed"(%2) {MANP = 2 : ui3} : (!FHE.eint<3>) -> !FHE.esint<3>
// CHECK-NEXT:     %4 = "FHE.apply_lookup_table"(%3, %cst) {MANP = 1 : ui1} : (!FHE.esint<3>, tensor<8xi64>) -> !FHE.eint<3>
// CHECK-NEXT:     %5 = "FHE.sub_eint"(%1, %4) {MANP = 2 : ui3} : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
// CHECK-NEXT:     return %5 : !FHE.eint<3>
// CHECK-NEXT: }
func.func @simple_eint(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<3>) -> !FHE.eint<3> {
  %0 = "FHE.mul_eint"(%arg0, %arg1): (!FHE.eint<3>, !FHE.eint<3>) -> (!FHE.eint<3>)
  return %0: !FHE.eint<3>
}
