// RUN: concretecompiler --split-input-file --action=roundtrip %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3xi3>, %[[a1:.*]]: tensor<3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3xi3>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<2>>, %[[a1:.*]]: tensor<3x2xi3>) -> tensor<2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<3x2xi3>) -> tensor<2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<2x!FHE.eint<2>>
  return %0 : tensor<2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3xi3>, %[[a1:.*]]: tensor<3x2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3xi3>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<2>>
  return %0 : tensor<2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<2>>, %[[a1:.*]]: tensor<4x3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<3x!FHE.eint<2>>, tensor<4x3x2xi3>) -> tensor<4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<4x3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<4x3x2xi3>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3xi3>, %[[a1:.*]]: tensor<4x3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<3xi3>, tensor<4x3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3xi3>, %y: tensor<4x3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<3xi3>, tensor<4x3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<2>>, %[[a1:.*]]: tensor<4x5x3x2xi3>) -> tensor<4x5x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<3x!FHE.eint<2>>, tensor<4x5x3x2xi3>) -> tensor<4x5x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x5x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<4x5x3x2xi3>) -> tensor<4x5x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<4x5x3x2xi3>) -> tensor<4x5x2x!FHE.eint<2>>
  return %0 : tensor<4x5x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3xi3>, %[[a1:.*]]: tensor<4x5x3x2x!FHE.eint<2>>) -> tensor<4x5x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<3xi3>, tensor<4x5x3x2x!FHE.eint<2>>) -> tensor<4x5x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x5x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3xi3>, %y: tensor<4x5x3x2x!FHE.eint<2>>) -> tensor<4x5x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<3xi3>, tensor<4x5x3x2x!FHE.eint<2>>) -> tensor<4x5x2x!FHE.eint<2>>
  return %0 : tensor<4x5x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<3xi3>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<4x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<3xi3>) -> tensor<4x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<4x!FHE.eint<2>>
  return %0 : tensor<4x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3xi3>, %[[a1:.*]]: tensor<3x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3xi3>, %y: tensor<3x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>>
  return %0 : tensor<4x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<3xi3>) -> tensor<2x4x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<2x4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<2x4x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<3xi3>) -> tensor<2x4x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<2x4x!FHE.eint<2>>
  return %0 : tensor<2x4x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3xi3>, %[[a1:.*]]: tensor<3x!FHE.eint<2>>) -> tensor<2x4x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<2x4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<2x4x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<3x!FHE.eint<2>>) -> tensor<2x4x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<2x4x!FHE.eint<2>>
  return %0 : tensor<2x4x!FHE.eint<2>>
}


// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<3xi3>) -> tensor<5x2x4x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<5x2x4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<5x2x4x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<3xi3>) -> tensor<5x2x4x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<5x2x4x!FHE.eint<2>>
  return %0 : tensor<5x2x4x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x2x4x3xi3>, %[[a1:.*]]: tensor<3x!FHE.eint<2>>) -> tensor<5x2x4x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<5x2x4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<5x2x4x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x2x4x3xi3>, %y: tensor<3x!FHE.eint<2>>) -> tensor<5x2x4x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x2x4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<5x2x4x!FHE.eint<2>>
  return %0 : tensor<5x2x4x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>>
  return %0 : tensor<2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3xi3>, %[[a1:.*]]: tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<2x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>>
  return %0 : tensor<2x4x2x!FHE.eint<2>>
}


// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<1x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<1x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<1x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<1x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>>
  return %0 : tensor<2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<1x4x3xi3>, %[[a1:.*]]: tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<1x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<1x4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<1x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>>
  return %0 : tensor<2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>>
  return %0 : tensor<2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3xi3>, %[[a1:.*]]: tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>>
  return %0 : tensor<2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<2x4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<2x4x2x!FHE.eint<2>>
  return %0 : tensor<2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3xi3>, %[[a1:.*]]: tensor<3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<2x4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<2x4x2x!FHE.eint<2>>
  return %0 : tensor<2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<5x2x4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x2x4x3xi3>, %[[a1:.*]]: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<5x2x4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x2x4x3xi3>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x2x4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<5x2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x2x4x3xi3>, %[[a1:.*]]: tensor<2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<5x2x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x2x4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x2x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<5x2x4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x2x4x3xi3>, %[[a1:.*]]: tensor<3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<5x2x4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x2x4x3xi3>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x2x4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<2x4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x4x3xi3>, %[[a1:.*]]: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<2x4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3xi3>, %[[a1:.*]]: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3xi3>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x1x4x3x!FHE.eint<2>>, %[[a1:.*]]: tensor<2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_eint_int"(%[[a0]], %[[a1]]) : (tensor<5x1x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x1x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x1x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x1x4x3xi3>, %[[a1:.*]]: tensor<2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.matmul_int_eint"(%[[a0]], %[[a1]]) : (tensor<5x1x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<5x2x4x2x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x1x4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x1x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<5x2x4x2x!FHE.eint<2>>
  return %0 : tensor<5x2x4x2x!FHE.eint<2>>
}
