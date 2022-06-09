// RUN: concretecompiler --action=dump-fhe %s 2>&1| FileCheck %s

// CHECK: func @add_eint_int_1D(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_int_1D(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[0, 0, 0, 0]> : tensor<4xi3>
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func @add_eint_int_1D_broadcast(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_int_1D_broadcast(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[0]> : tensor<1xi3>
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<1xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func @add_eint_int_2D_broadcast(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x3x!FHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_int_2D_broadcast(%a0: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
  %a1 = arith.constant dense<[[0]]> : tensor<1x1xi3>
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x3x!FHE.eint<2>>, tensor<1x1xi3>) -> tensor<4x3x!FHE.eint<2>>
  return %1: tensor<4x3x!FHE.eint<2>>
}

// CHECK: func @mul_eint_int_1D(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func @mul_eint_int_1D(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[1, 1, 1, 1]> : tensor<4xi3>
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func @mul_eint_int_1D_broadcast(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func @mul_eint_int_1D_broadcast(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[1]> : tensor<1xi3>
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<1xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func @mul_eint_int_2D_broadcast(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x3x!FHE.eint<2>>
// CHECK-NEXT: }
func @mul_eint_int_2D_broadcast(%a0: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
  %a1 = arith.constant dense<[[1]]> : tensor<1x1xi3>
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x3x!FHE.eint<2>>, tensor<1x1xi3>) -> tensor<4x3x!FHE.eint<2>>
  return %1: tensor<4x3x!FHE.eint<2>>
}
