// RUN: concretecompiler --action=dump-tfhe --passes EncryptedMulToDoubleTLU --split-input-file %s 2>&1 | FileCheck %s

// CHECK:      func.func @simple_eint(%[[a0:.*]]: !FHE.eint<3>, %[[a1:.*]]: !FHE.eint<3>) -> !FHE.eint<3> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.add_eint"(%[[a0]], %[[a1]]) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
// CHECK-NEXT:   %[[v1:.*]] = arith.constant dense<[0, 0, 1, 2, 4, 6, 9, 12]> : tensor<8xi64>
// CHECK-NEXT:   %[[v2:.*]] = "FHE.apply_lookup_table"(%[[v0]], %[[v1]]) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<3>
// CHECK-NEXT:   %[[v3:.*]] = "FHE.sub_eint"(%[[a0]], %[[a1]]) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
// CHECK-NEXT:   %[[v4:.*]] = "FHE.to_signed"(%[[v3]]) : (!FHE.eint<3>) -> !FHE.esint<3>
// CHECK-NEXT:   %[[v5:.*]] = arith.constant dense<[0, 0, 1, 2, 4, 2, 1, 0]> : tensor<8xi64>
// CHECK-NEXT:   %[[v6:.*]] = "FHE.apply_lookup_table"(%[[v4]], %[[v5]]) : (!FHE.esint<3>, tensor<8xi64>) -> !FHE.eint<3>
// CHECK-NEXT:   %[[v7:.*]] = "FHE.sub_eint"(%[[v2]], %[[v6]]) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
// CHECK-NEXT:   return %[[v7]] : !FHE.eint<3>
// CHECK-NEXT: }
func.func @simple_eint(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<3>) -> !FHE.eint<3> {
  %0 = "FHE.mul_eint"(%arg0, %arg1): (!FHE.eint<3>, !FHE.eint<3>) -> (!FHE.eint<3>)
  return %0: !FHE.eint<3>
}
