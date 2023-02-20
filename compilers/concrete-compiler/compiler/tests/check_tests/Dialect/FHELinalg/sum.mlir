// RUN: concretecompiler --split-input-file --action=roundtrip %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<0x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [], keep_dims = false} : (tensor<0x!FHE.eint<7>>) -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<0x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<0x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [], keep_dims = false} : (tensor<3x0x4x!FHE.eint<7>>) -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<3x0x4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [1], keep_dims = false} : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1] } : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>>
  return %0 : tensor<3x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x1x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [1], keep_dims = true} : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x1x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x1x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x1x4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1], keep_dims = true } : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x1x4x!FHE.eint<7>>
  return %0 : tensor<3x1x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [2], keep_dims = false} : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x0x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [2] } : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x!FHE.eint<7>>
  return %0 : tensor<3x0x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [2], keep_dims = true} : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x0x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [2], keep_dims = true } : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x1x!FHE.eint<7>>
  return %0 : tensor<3x0x1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [], keep_dims = false} : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0], keep_dims = false} : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0] } : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [], keep_dims = true} : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { keep_dims = true } : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
  return %0 : tensor<1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0], keep_dims = true} : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0], keep_dims = true } : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
  return %0 : tensor<1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [], keep_dims = false} : (tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [], keep_dims = true} : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0], keep_dims = false} : (tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0] } : (tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
  return %0 : tensor<4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0], keep_dims = true} : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>>
  return %0 : tensor<1x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [1], keep_dims = false} : (tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1] } : (tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>
  return %0 : tensor<3x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [1], keep_dims = true} : (tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>>
  return %0 : tensor<3x1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0, 1], keep_dims = false} : (tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 1] } : (tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0, 1], keep_dims = true} : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 1], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [], keep_dims = false} : (tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [], keep_dims = true} : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [1], keep_dims = false} : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x2x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>>
  return %0 : tensor<3x2x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [1], keep_dims = true} : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x1x2x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>>
  return %0 : tensor<3x1x2x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0, 2], keep_dims = false} : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 2] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
  return %0 : tensor<4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0, 2], keep_dims = true} : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x4x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 2], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>>
  return %0 : tensor<1x4x1x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0, 1, 2], keep_dims = false} : (tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 2] } : (tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.sum"(%[[a0]]) {axes = [0, 1, 2], keep_dims = true} : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 2], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x1x!FHE.eint<7>>
}
