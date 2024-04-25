// RUN: concretecompiler --action=dump-fhe --optimizer-strategy=V0 --skip-program-info %s 2>&1| FileCheck %s

// CHECK: func.func @add_eint_int_1D(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @add_eint_int_1D(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[0, 0, 0, 0]> : tensor<4xi3>
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func.func @add_eint_int_1D_broadcast(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @add_eint_int_1D_broadcast(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[0]> : tensor<1xi3>
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<1xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func.func @add_eint_int_2D_broadcast(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x3x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @add_eint_int_2D_broadcast(%a0: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
  %a1 = arith.constant dense<[[0]]> : tensor<1x1xi3>
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x3x!FHE.eint<2>>, tensor<1x1xi3>) -> tensor<4x3x!FHE.eint<2>>
  return %1: tensor<4x3x!FHE.eint<2>>
}

// CHECK: func.func @sub_eint_int_1D(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @sub_eint_int_1D(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[0, 0, 0, 0]> : tensor<4xi3>
  %1 = "FHELinalg.sub_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func.func @sub_eint_int_1D_broadcast(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @sub_eint_int_1D_broadcast(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[0]> : tensor<1xi3>
  %1 = "FHELinalg.sub_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<1xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func.func @sub_eint_int_2D_broadcast(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x3x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @sub_eint_int_2D_broadcast(%a0: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
  %a1 = arith.constant dense<[[0]]> : tensor<1x1xi3>
  %1 = "FHELinalg.sub_eint_int"(%a0, %a1) : (tensor<4x3x!FHE.eint<2>>, tensor<1x1xi3>) -> tensor<4x3x!FHE.eint<2>>
  return %1: tensor<4x3x!FHE.eint<2>>
}

// CHECK: func.func @mul_eint_int_1D(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @mul_eint_int_1D(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[1, 1, 1, 1]> : tensor<4xi3>
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func.func @mul_eint_int_1D_broadcast(%[[a0:.*]]: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @mul_eint_int_1D_broadcast(%a0: tensor<4x!FHE.eint<2>>) -> tensor<4x!FHE.eint<2>> {
  %a1 = arith.constant dense<[1]> : tensor<1xi3>
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<2>>, tensor<1xi3>) -> tensor<4x!FHE.eint<2>>
  return %1: tensor<4x!FHE.eint<2>>
}

// CHECK: func.func @mul_eint_int_2D_broadcast(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
// CHECK-NEXT: return %[[a0]] : tensor<4x3x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @mul_eint_int_2D_broadcast(%a0: tensor<4x3x!FHE.eint<2>>) -> tensor<4x3x!FHE.eint<2>> {
  %a1 = arith.constant dense<[[1]]> : tensor<1x1xi3>
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x3x!FHE.eint<2>>, tensor<1x1xi3>) -> tensor<4x3x!FHE.eint<2>>
  return %1: tensor<4x3x!FHE.eint<2>>
}

// CHECK-LABEL: func.func @round(%arg0: tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>>
func.func @round(%arg0: tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>> {
  // CHECK-NEXT: return %arg0 : tensor<4x!FHE.eint<5>>

  %1 = "FHELinalg.round"(%arg0) : (tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>>
  return %1: tensor<4x!FHE.eint<5>>
}

// CHECK: func.func @mul_by_clear_zero(%[[a0:.*]]: tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>>
func.func @mul_by_clear_zero(%arg0: tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>> {
  // CHECK-NEXT: %[[v0:.*]] = "FHE.zero_tensor"()
  // CHECK-NEXT: return %[[v0]] : tensor<4x!FHE.eint<5>>
  %cst_0 = arith.constant dense<0> : tensor<4xi5>
  %1 = "FHELinalg.mul_eint_int"(%arg0, %cst_0) : (tensor<4x!FHE.eint<5>>, tensor<4xi5>) -> tensor<4x!FHE.eint<5>>
  return %1: tensor<4x!FHE.eint<5>>
}

// CHECK: func.func @mul_by_encrypted_zero(%[[a0:.*]]: tensor<4xi5>) -> tensor<4x!FHE.eint<5>>
func.func @mul_by_encrypted_zero(%arg0: tensor<4xi5>) -> tensor<4x!FHE.eint<5>> {
  // CHECK-NEXT: %[[v0:.*]] = "FHE.zero_tensor"()
  // CHECK-NEXT: return %[[v0]] : tensor<4x!FHE.eint<5>>
  %cst0 = "FHE.zero_tensor"() : () -> tensor<4x!FHE.eint<5>>
  %1 = "FHELinalg.mul_eint_int"(%cst0, %arg0) : (tensor<4x!FHE.eint<5>>, tensor<4xi5>) -> tensor<4x!FHE.eint<5>>
  return %1: tensor<4x!FHE.eint<5>>
}

// CHECK: func.func @matmul_eint_int_clear_zero(%[[a0:.*]]: tensor<4x3x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
// CHECK-NEXT: %[[v0:.*]] = "FHE.zero_tensor"()
// CHECK-NEXT: return %[[v0]]
func.func @matmul_eint_int_clear_zero(%x: tensor<4x3x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
  %y = arith.constant dense<0> : tensor<3x2xi3>
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// CHECK: func.func @matmul_eint_int_encrypted_zero(%[[a0:.*]]: tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
// CHECK-NEXT: %[[v0:.*]] = "FHE.zero_tensor"()
// CHECK-NEXT: return %[[v0]]
func.func @matmul_eint_int_encrypted_zero(%y: tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
  %x = "FHE.zero_tensor"() : () -> tensor<4x3x!FHE.eint<2>>
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// CHECK:      func.func @matmul_int_eint_clear_zero(%[[a0:.*]]: tensor<3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
// CHECK-NEXT: %[[v0:.*]] = "FHE.zero_tensor"()
// CHECK-NEXT: return %[[v0]]
func.func @matmul_int_eint_clear_zero(%y: tensor<3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
  %x = arith.constant dense<0> : tensor<4x3xi3>
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// CHECK: func.func @matmul_int_eint_encrypted_zero(%[[a0:.*]]: tensor<4x3xi3>) -> tensor<4x2x!FHE.eint<2>> {
// CHECK-NEXT: %[[v0:.*]] = "FHE.zero_tensor"()
// CHECK-NEXT: return %[[v0]]
func.func @matmul_int_eint_encrypted_zero(%x: tensor<4x3xi3>) -> tensor<4x2x!FHE.eint<2>> {
  %y = "FHE.zero_tensor"() : () -> tensor<3x2x!FHE.eint<2>>
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// CHECK: func.func @to_signed_zero() -> tensor<4x!FHE.esint<7>> {
// CHECK-NEXT: %[[v0:.*]] = "FHE.zero_tensor"()
// CHECK-NEXT: return %[[v0]]
func.func @to_signed_zero() -> tensor<4x!FHE.esint<7>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x!FHE.eint<7>>
  %1 = "FHELinalg.to_signed"(%0) : (tensor<4x!FHE.eint<7>>) -> tensor<4x!FHE.esint<7>>
  return %1 : tensor<4x!FHE.esint<7>>
}

// CHECK: func.func @to_unsigned_zero() -> tensor<4x!FHE.eint<7>> {
// CHECK-NEXT: %[[v0:.*]] = "FHE.zero_tensor"()
// CHECK-NEXT: return %[[v0]]
func.func @to_unsigned_zero() -> tensor<4x!FHE.eint<7>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x!FHE.esint<7>>
  %1 = "FHELinalg.to_unsigned"(%0) : (tensor<4x!FHE.esint<7>>) -> tensor<4x!FHE.eint<7>>
  return %1 : tensor<4x!FHE.eint<7>>
}

// CHECK: func.func @concat_1_operand(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
// CHECK-NEXT: return %[[a0]]
func.func @concat_1_operand(%x: tensor<4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
  %0 = "FHELinalg.concat"(%x) : (tensor<4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
  return %0 : tensor<4x!FHE.eint<7>>
}
