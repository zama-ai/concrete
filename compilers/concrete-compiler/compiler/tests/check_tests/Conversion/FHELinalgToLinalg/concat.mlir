// RUN: concretecompiler %s --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<7x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0] [3] [1] : tensor<3x!FHE.eint<7>> into tensor<7x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][3] [4] [1] : tensor<4x!FHE.eint<7>> into tensor<7x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<7x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<7>>, %y: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<3x!FHE.eint<7>>, tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<7x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0] [3] [1] : tensor<3x!FHE.eint<7>> into tensor<7x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][3] [4] [1] : tensor<4x!FHE.eint<7>> into tensor<7x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<7x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<7>>, %y: tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<3x!FHE.eint<7>>, tensor<4x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0, 0] [3, 4] [1, 1] : tensor<3x4x!FHE.eint<7>> into tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][3, 0] [4, 4] [1, 1] : tensor<4x4x!FHE.eint<7>> into tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<3x4x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>>
  return %0 : tensor<7x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0, 0] [3, 4] [1, 1] : tensor<3x4x!FHE.eint<7>> into tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][3, 0] [4, 4] [1, 1] : tensor<4x4x!FHE.eint<7>> into tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<7x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<3x4x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<7x4x!FHE.eint<7>>
  return %0 : tensor<7x4x!FHE.eint<7>>
}


// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x3x!FHE.eint<7>>, %[[a1:.*]]: tensor<4x4x!FHE.eint<7>>) -> tensor<4x7x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<4x7x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0, 0] [4, 3] [1, 1] : tensor<4x3x!FHE.eint<7>> into tensor<4x7x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][0, 3] [4, 4] [1, 1] : tensor<4x4x!FHE.eint<7>> into tensor<4x7x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<4x7x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x3x!FHE.eint<7>>, %y: tensor<4x4x!FHE.eint<7>>) -> tensor<4x7x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<4x3x!FHE.eint<7>>, tensor<4x4x!FHE.eint<7>>) -> tensor<4x7x!FHE.eint<7>>
  return %0 : tensor<4x7x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0, 0, 0] [2, 3, 4] [1, 1, 1] : tensor<2x3x4x!FHE.eint<7>> into tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][2, 0, 0] [2, 3, 4] [1, 1, 1] : tensor<2x3x4x!FHE.eint<7>> into tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x3x4x!FHE.eint<7>>, %y: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>>
  return %0 : tensor<4x3x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0, 0, 0] [2, 3, 4] [1, 1, 1] : tensor<2x3x4x!FHE.eint<7>> into tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][2, 0, 0] [2, 3, 4] [1, 1, 1] : tensor<2x3x4x!FHE.eint<7>> into tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<4x3x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x3x4x!FHE.eint<7>>, %y: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x4x!FHE.eint<7>>
  return %0 : tensor<4x3x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x6x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<2x6x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0, 0, 0] [2, 3, 4] [1, 1, 1] : tensor<2x3x4x!FHE.eint<7>> into tensor<2x6x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][0, 3, 0] [2, 3, 4] [1, 1, 1] : tensor<2x3x4x!FHE.eint<7>> into tensor<2x6x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<2x6x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x3x4x!FHE.eint<7>>, %y: tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x6x4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x6x4x!FHE.eint<7>>
  return %0 : tensor<2x6x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>, %[[a1:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x3x8x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.empty() : tensor<2x3x8x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.insert_slice %[[a0]] into %[[v0]][0, 0, 0] [2, 3, 4] [1, 1, 1] : tensor<2x3x4x!FHE.eint<7>> into tensor<2x3x8x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = tensor.insert_slice %[[a1]] into %[[v1]][0, 0, 4] [2, 3, 4] [1, 1, 1] : tensor<2x3x4x!FHE.eint<7>> into tensor<2x3x8x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : tensor<2x3x8x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x3x4x!FHE.eint<7>>, %y: tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x3x8x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x, %y) { axis = 2 } : (tensor<2x3x4x!FHE.eint<7>>, tensor<2x3x4x!FHE.eint<7>>) -> tensor<2x3x8x!FHE.eint<7>>
  return %0 : tensor<2x3x8x!FHE.eint<7>>
}
