// RUN: concretecompiler --split-input-file --action=roundtrip --verify-diagnostics %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<3x!FHE.eint<7>>, tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<7x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<7>>, %y: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<3x!FHE.eint<7>>, tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<3x!FHE.eint<7>>, tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<7x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<7>>, %y: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<3x!FHE.eint<7>>, tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<3x4x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<3x4x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>>
  return %0 : tensor<7x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<3x4x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<3x4x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>>
  return %0 : tensor<7x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x4x!FHE.eint<7>>) -> tensor<4x7x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<4x3x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<4x7x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x7x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3x!FHE.eint<7>>, %y: tensor<4x4x!FHE.eint<7>>) -> tensor<4x7x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<4x3x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<4x7x!FHE.eint<7>>
  return %0 : tensor<4x7x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x3x4x!FHE.eint<7>>, %y: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>>
  return %0 : tensor<4x3x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x3x4x!FHE.eint<7>>, %y: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>>
  return %0 : tensor<4x3x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x6x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x6x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x6x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x3x4x!FHE.eint<7>>, %y: tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x6x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x6x4x!FHE.eint<7>>
  return %0 : tensor<2x6x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x3x8x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.concat"(%[[a0]], %[[a1]]) : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x3x8x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x3x8x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x3x4x!FHE.eint<7>>, %y: tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x3x8x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 2 } : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x3x8x!FHE.eint<7>>
  return %0 : tensor<2x3x8x!FHE.eint<7>>
}
