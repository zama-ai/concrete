// RUN: concretecompiler --split-input-file --action=dump-tfhe --passes fhe-max-transform %s 2>&1 | FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: !FHE.eint<5>, %[[a1:.*]]: !FHE.eint<5>) -> !FHE.eint<5> {
// CHECK-NEXT:   %[[v0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<32xi64>
// CHECK-NEXT:   %[[v1:.*]] = "FHE.to_signed"(%[[a0]]) : (!FHE.eint<5>) -> !FHE.esint<5>
// CHECK-NEXT:   %[[v2:.*]] = "FHE.to_signed"(%[[a1]]) : (!FHE.eint<5>) -> !FHE.esint<5>
// CHECK-NEXT:   %[[v3:.*]] = "FHE.sub_eint"(%[[v1]], %[[v2]]) : (!FHE.esint<5>, !FHE.esint<5>) -> !FHE.esint<5>
// CHECK-NEXT:   %[[v4:.*]] = "FHE.apply_lookup_table"(%[[v3]], %[[v0]]) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<5>
// CHECK-NEXT:   %[[v5:.*]] = "FHE.add_eint"(%[[v4]], %[[a1]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:   return %[[v5]] : !FHE.eint<5>
// CHECK-NEXT: }
func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
  %0 = "FHE.max_eint"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
  return %0 : !FHE.eint<5>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: !FHE.esint<5>, %[[a1:.*]]: !FHE.esint<5>) -> !FHE.esint<5> {
// CHECK-NEXT:   %[[v0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<32xi64>
// CHECK-NEXT:   %[[v1:.*]] = "FHE.sub_eint"(%[[a0]], %[[a1]]) : (!FHE.esint<5>, !FHE.esint<5>) -> !FHE.esint<5>
// CHECK-NEXT:   %[[v2:.*]] = "FHE.apply_lookup_table"(%[[v1]], %[[v0]]) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.esint<5>
// CHECK-NEXT:   %[[v3:.*]] = "FHE.add_eint"(%[[v2]], %[[a1]]) : (!FHE.esint<5>, !FHE.esint<5>) -> !FHE.esint<5>
// CHECK-NEXT:   return %[[v3:.*]] : !FHE.esint<5>
// CHECK-NEXT: }
func.func @main(%arg0: !FHE.esint<5>, %arg1: !FHE.esint<5>) -> !FHE.esint<5> {
  %0 = "FHE.max_eint"(%arg0, %arg1) : (!FHE.esint<5>, !FHE.esint<5>) -> !FHE.esint<5>
  return %0 : !FHE.esint<5>
}
