// RUN: concretecompiler --split-input-file --action=roundtrip %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.round"(%[[a0]]) : (tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.eint<6>> {
  %0 = "FHELinalg.round"(%arg0) : (tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.eint<6>>
  return %0 : tensor<5x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3x2x!FHE.eint<8>>) -> tensor<4x3x2x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.round"(%[[a0]]) : (tensor<4x3x2x!FHE.eint<8>>) -> tensor<4x3x2x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x3x2x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x3x2x!FHE.eint<8>>) -> tensor<4x3x2x!FHE.eint<6>> {
  %0 = "FHELinalg.round"(%arg0) : (tensor<4x3x2x!FHE.eint<8>>) -> tensor<4x3x2x!FHE.eint<6>>
  return %0 : tensor<4x3x2x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x!FHE.esint<8>>) -> tensor<5x!FHE.esint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.round"(%[[a0]]) : (tensor<5x!FHE.esint<8>>) -> tensor<5x!FHE.esint<6>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x!FHE.esint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<5x!FHE.esint<8>>) -> tensor<5x!FHE.esint<6>> {
  %0 = "FHELinalg.round"(%arg0) : (tensor<5x!FHE.esint<8>>) -> tensor<5x!FHE.esint<6>>
  return %0 : tensor<5x!FHE.esint<6>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3x2x!FHE.esint<8>>) -> tensor<4x3x2x!FHE.esint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.round"(%[[a0]]) : (tensor<4x3x2x!FHE.esint<8>>) -> tensor<4x3x2x!FHE.esint<6>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x3x2x!FHE.esint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x3x2x!FHE.esint<8>>) -> tensor<4x3x2x!FHE.esint<6>> {
  %0 = "FHELinalg.round"(%arg0) : (tensor<4x3x2x!FHE.esint<8>>) -> tensor<4x3x2x!FHE.esint<6>>
  return %0 : tensor<4x3x2x!FHE.esint<6>>
}
