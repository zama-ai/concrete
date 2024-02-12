// RUN: concretecompiler --split-input-file --action=dump-fhe --passes canonicalize %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<5> {
// CHECK-NEXT:   %cst = arith.constant dense<[0, 0, 2, 4, 8, 12, 18, 24]> : tensor<8xi64>
// CHECK-NEXT:   %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<5>
// CHECK-NEXT:   return %0 : !FHE.eint<5>
// CHECK-NEXT: }
func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<5> {
  %c2_i3 = arith.constant 2 : i3
  %cst = arith.constant dense<[0, 1, 4, 9, 16, 25, 36, 49]> : tensor<8xi64>
  %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<6>
  %cst_0 = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31]> : tensor<64xi64>
  %1 = "FHE.apply_lookup_table"(%0, %cst_0) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<5>
  return %1 : !FHE.eint<5>
}

// -----

// CHECK:      func.func @main(%arg0: !FHE.eint<2>) -> !FHE.eint<4> {
// CHECK-NEXT:   %cst = arith.constant dense<[0, 0, 2, 4]> : tensor<4xi64>
// CHECK-NEXT:   %cst_0 = arith.constant dense<[0, 1, 4, 9]> : tensor<4xi64>
// CHECK-NEXT:   %0 = "FHE.apply_lookup_table"(%arg0, %cst_0) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<4>
// CHECK-NEXT:   %1 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<4>
// CHECK-NEXT:   %2 = "FHE.add_eint"(%1, %0) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
// CHECK-NEXT:   return %2 : !FHE.eint<4>
// CHECK-NEXT: }
func.func @main(%arg0: !FHE.eint<2>) -> !FHE.eint<4> {
  %c2_i3 = arith.constant 2 : i3
  %cst = arith.constant dense<[0, 1, 4, 9]> : tensor<4xi64>
  %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<4>
  %cst_0 = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xi64>
  %1 = "FHE.apply_lookup_table"(%0, %cst_0) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
  %2 = "FHE.add_eint"(%1, %0) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
  return %2 : !FHE.eint<4>
}

// -----

// CHECK:      func.func @main(%arg0: !FHE.eint<6>, %arg1: !FHE.eint<4>) -> !FHE.eint<4> {
// CHECK-NEXT:   %cst = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xi64>
// CHECK-NEXT:   %cst_0 = arith.constant dense<[0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8]> : tensor<64xi64>
// CHECK-NEXT:   %0 = "FHE.apply_lookup_table"(%arg0, %cst_0) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<4>
// CHECK-NEXT:   %1 = "FHE.apply_lookup_table"(%0, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
// CHECK-NEXT:   %2 = "FHE.add_eint"(%0, %arg1) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
// CHECK-NEXT:   %3 = "FHE.add_eint"(%1, %2) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
// CHECK-NEXT:   return %3 : !FHE.eint<4>
// CHECK-NEXT: }
func.func @main(%arg0: !FHE.eint<6>, %arg1: !FHE.eint<4>) -> !FHE.eint<4> {
  %cst = arith.constant dense<[0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8]> : tensor<64xi64>
  %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<4>
  %c2_i3 = arith.constant 2 : i3
  %cst_0 = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xi64>
  %1 = "FHE.apply_lookup_table"(%0, %cst_0) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
  %2 = "FHE.add_eint"(%0, %arg1) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
  %3 = "FHE.add_eint"(%1, %2) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
  return %3 : !FHE.eint<4>
}

// -----

// CHECK:      func.func @main(%arg0: !FHE.esint<5>) -> !FHE.esint<8> {
// CHECK-NEXT:   %cst = arith.constant dense<[0, 0, 1, 1, 8, 8, 27, 27, 64, 64, 125, 125, 216, 216, 343, 343, -512, -512, -343, -343, -216, -216, -125, -125, -64, -64, -27, -27, -8, -8, -1, -1]> : tensor<32xi64>
// CHECK-NEXT:   %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.esint<8>
// CHECK-NEXT:   return %0 : !FHE.esint<8>
// CHECK-NEXT: }
func.func @main(%arg0: !FHE.esint<5>) -> !FHE.esint<8> {
  %c2_i3 = arith.constant 2 : i3
  %cst = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, -8, -8, -7, -7, -6, -6, -5, -5, -4, -4, -3, -3, -2, -2, -1, -1]> : tensor<32xi64>
  %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.esint<4>
  %c3_i3 = arith.constant 3 : i3
  %cst_0 = arith.constant dense<[0, 1, 8, 27, 64, 125, 216, 343, -512, -343, -216, -125, -64, -27, -8, -1]> : tensor<16xi64>
  %1 = "FHE.apply_lookup_table"(%0, %cst_0) : (!FHE.esint<4>, tensor<16xi64>) -> !FHE.esint<8>
  return %1 : !FHE.esint<8>
}
