// RUN: zamacompiler --entry-dialect=hlfhe --action=roundtrip %s 2>&1| FileCheck %s

/////////////////////////////////////////////////
// HLFHELinalg.add_eint_int
/////////////////////////////////////////////////

// 1D tensor
// CHECK: func @add_eint_int_1D(%[[a0:.*]]: tensor<4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<4xi3>) -> tensor<4x!HLFHE.eint<2>> {
// CHECK-NEXT: %[[V0:.*]] = "HLFHELinalg.add_eint_int"(%[[a0]], %[[a1]]) : (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: return %[[V0]] : tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_int_1D(%a0: tensor<4x!HLFHE.eint<2>>, %a1: tensor<4xi3>) -> tensor<4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> tensor<4x!HLFHE.eint<2>>
  return %1: tensor<4x!HLFHE.eint<2>>
}

// 2D tensor
// CHECK: func @add_eint_int_2D(%[[a0:.*]]: tensor<2x4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<2x4xi3>) -> tensor<2x4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.add_eint_int"(%[[a0]], %[[a1]]) : (tensor<2x4x!HLFHE.eint<2>>, tensor<2x4xi3>) -> tensor<2x4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<2x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_int_2D(%a0: tensor<2x4x!HLFHE.eint<2>>, %a1: tensor<2x4xi3>) -> tensor<2x4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x4x!HLFHE.eint<2>>, tensor<2x4xi3>) -> tensor<2x4x!HLFHE.eint<2>>
  return %1: tensor<2x4x!HLFHE.eint<2>>
}

// 10D tensor
// CHECK: func @add_eint_int_10D(%[[a0:.*]]: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<1x2x3x4x5x6x7x8x9x10xi3>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.add_eint_int"(%[[a0]], %[[a1]]) : (tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, tensor<1x2x3x4x5x6x7x8x9x10xi3>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_int_10D(%a0: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, %a1: tensor<1x2x3x4x5x6x7x8x9x10xi3>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, tensor<1x2x3x4x5x6x7x8x9x10xi3>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
  return %1: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
}

// Broadcasting with tensor with dimensions equals to one
// CHECK: func @add_eint_int_broadcast_1(%[[a0:.*]]: tensor<1x4x5x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<3x4x1xi3>) -> tensor<3x4x5x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.add_eint_int"(%[[a0]], %[[a1]]) : (tensor<1x4x5x!HLFHE.eint<2>>, tensor<3x4x1xi3>) -> tensor<3x4x5x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<3x4x5x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_int_broadcast_1(%a0: tensor<1x4x5x!HLFHE.eint<2>>, %a1: tensor<3x4x1xi3>) -> tensor<3x4x5x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<1x4x5x!HLFHE.eint<2>>, tensor<3x4x1xi3>) -> tensor<3x4x5x!HLFHE.eint<2>>
  return %1: tensor<3x4x5x!HLFHE.eint<2>>
}

// Broadcasting with a tensor less dimensions of another
// CHECK: func @add_eint_int_broadcast_2(%[[a0:.*]]: tensor<4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<3x4xi3>) -> tensor<3x4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.add_eint_int"(%[[a0]], %[[a1]]) : (tensor<4x!HLFHE.eint<2>>, tensor<3x4xi3>) -> tensor<3x4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<3x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_int_broadcast_2(%a0: tensor<4x!HLFHE.eint<2>>, %a1: tensor<3x4xi3>) -> tensor<3x4x!HLFHE.eint<2>> {
 %1 ="HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<2>>, tensor<3x4xi3>) -> tensor<3x4x!HLFHE.eint<2>>
 return %1: tensor<3x4x!HLFHE.eint<2>>
}

/////////////////////////////////////////////////
// HLFHELinalg.add_eint
/////////////////////////////////////////////////

// 1D tensor
// CHECK: func @add_eint_1D(%[[a0:.*]]: tensor<4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>> {
// CHECK-NEXT: %[[V0:.*]] = "HLFHELinalg.add_eint"(%[[a0]], %[[a1]]) : (tensor<4x!HLFHE.eint<2>>, tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: return %[[V0]] : tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_1D(%a0: tensor<4x!HLFHE.eint<2>>, %a1: tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<4x!HLFHE.eint<2>>, tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>>
  return %1: tensor<4x!HLFHE.eint<2>>
}

// 2D tensor
// CHECK: func @add_eint_2D(%[[a0:.*]]: tensor<2x4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<2x4x!HLFHE.eint<2>>) -> tensor<2x4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.add_eint"(%[[a0]], %[[a1]]) : (tensor<2x4x!HLFHE.eint<2>>, tensor<2x4x!HLFHE.eint<2>>) -> tensor<2x4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<2x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_2D(%a0: tensor<2x4x!HLFHE.eint<2>>, %a1: tensor<2x4x!HLFHE.eint<2>>) -> tensor<2x4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<2x4x!HLFHE.eint<2>>, tensor<2x4x!HLFHE.eint<2>>) -> tensor<2x4x!HLFHE.eint<2>>
  return %1: tensor<2x4x!HLFHE.eint<2>>
}

// 10D tensor
// CHECK: func @add_eint_10D(%[[a0:.*]]: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.add_eint"(%[[a0]], %[[a1]]) : (tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_10D(%a0: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, %a1: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
  return %1: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
}

// Broadcasting with tensor with dimensions equals to one
// CHECK: func @add_eint_broadcast_1(%[[a0:.*]]: tensor<1x4x5x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<3x4x1x!HLFHE.eint<2>>) -> tensor<3x4x5x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.add_eint"(%[[a0]], %[[a1]]) : (tensor<1x4x5x!HLFHE.eint<2>>, tensor<3x4x1x!HLFHE.eint<2>>) -> tensor<3x4x5x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<3x4x5x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_broadcast_1(%a0: tensor<1x4x5x!HLFHE.eint<2>>, %a1: tensor<3x4x1x!HLFHE.eint<2>>) -> tensor<3x4x5x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<1x4x5x!HLFHE.eint<2>>, tensor<3x4x1x!HLFHE.eint<2>>) -> tensor<3x4x5x!HLFHE.eint<2>>
  return %1: tensor<3x4x5x!HLFHE.eint<2>>
}

// Broadcasting with a tensor less dimensions of another
// CHECK: func @add_eint_broadcast_2(%[[a0:.*]]: tensor<4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<3x4x!HLFHE.eint<2>>) -> tensor<3x4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.add_eint"(%[[a0]], %[[a1]]) : (tensor<4x!HLFHE.eint<2>>, tensor<3x4x!HLFHE.eint<2>>) -> tensor<3x4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<3x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @add_eint_broadcast_2(%a0: tensor<4x!HLFHE.eint<2>>, %a1: tensor<3x4x!HLFHE.eint<2>>) -> tensor<3x4x!HLFHE.eint<2>> {
 %1 ="HLFHELinalg.add_eint"(%a0, %a1) : (tensor<4x!HLFHE.eint<2>>, tensor<3x4x!HLFHE.eint<2>>) -> tensor<3x4x!HLFHE.eint<2>>
 return %1: tensor<3x4x!HLFHE.eint<2>>
}