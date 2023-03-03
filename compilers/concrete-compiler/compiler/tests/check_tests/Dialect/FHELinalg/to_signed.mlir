// RUN: concretecompiler --split-input-file --action=roundtrip %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<2>>) -> tensor<3x!FHE.esint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.to_signed"(%[[a0]]) : (tensor<3x!FHE.eint<2>>) -> tensor<3x!FHE.esint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x!FHE.esint<2>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x!FHE.eint<2>>) -> tensor<3x!FHE.esint<2>> {
  %1 = "FHELinalg.to_signed"(%arg0): (tensor<3x!FHE.eint<2>>) -> tensor<3x!FHE.esint<2>>
  return %1 : tensor<3x!FHE.esint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.esint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.to_signed"(%[[a0]]) : (tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.esint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x2x!FHE.esint<2>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.esint<2>> {
  %1 = "FHELinalg.to_signed"(%arg0): (tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.esint<2>>
  return %1 : tensor<3x2x!FHE.esint<2>>
}
