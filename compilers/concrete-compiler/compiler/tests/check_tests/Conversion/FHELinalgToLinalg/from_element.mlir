// RUN: concretecompiler %s --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: !FHE.eint<7>) -> tensor<1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.from_elements %[[a0]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: !FHE.eint<7>) -> tensor<1x!FHE.eint<7>> {
  %1 = "FHELinalg.from_element"(%arg0) : (!FHE.eint<7>) -> tensor<1x!FHE.eint<7>>
  return %1 : tensor<1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: i8) -> tensor<1xi8> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.from_elements %[[a0]] : tensor<1xi8>
// CHECK-NEXT:   return %[[v0]] : tensor<1xi8>
// CHECK-NEXT: }
func.func @main(%arg0: i8) -> tensor<1xi8> {
  %1 = "FHELinalg.from_element"(%arg0) : (i8) -> tensor<1xi8>
  return %1 : tensor<1xi8>
}
