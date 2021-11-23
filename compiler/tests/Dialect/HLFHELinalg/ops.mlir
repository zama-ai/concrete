// RUN: zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

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


/////////////////////////////////////////////////
// HLFHELinalg.sub_eint_int
/////////////////////////////////////////////////

// 1D tensor
// CHECK: func @sub_int_eint_1D(%[[a0:.*]]: tensor<4xi3>, %[[a1:.*]]: tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>> {
// CHECK-NEXT: %[[V0:.*]] = "HLFHELinalg.sub_int_eint"(%[[a0]], %[[a1]]) : (tensor<4xi3>, tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: return %[[V0]] : tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @sub_int_eint_1D(%a0: tensor<4xi3>, %a1: tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4xi3>, tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>>
  return %1: tensor<4x!HLFHE.eint<2>>
}

// 2D tensor
// CHECK: func @sub_int_eint_2D(%[[a0:.*]]: tensor<2x4xi3>, %[[a1:.*]]: tensor<2x4x!HLFHE.eint<2>>) -> tensor<2x4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.sub_int_eint"(%[[a0]], %[[a1]]) : (tensor<2x4xi3>, tensor<2x4x!HLFHE.eint<2>>) -> tensor<2x4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<2x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @sub_int_eint_2D(%a0: tensor<2x4xi3>, %a1: tensor<2x4x!HLFHE.eint<2>>) -> tensor<2x4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<2x4xi3>, tensor<2x4x!HLFHE.eint<2>>) -> tensor<2x4x!HLFHE.eint<2>>
  return %1: tensor<2x4x!HLFHE.eint<2>>
}

// 10D tensor
// CHECK: func @sub_int_eint_10D(%[[a0:.*]]: tensor<1x2x3x4x5x6x7x8x9x10xi3>, %[[a1:.*]]: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.sub_int_eint"(%[[a0]], %[[a1]]) : (tensor<1x2x3x4x5x6x7x8x9x10xi3>, tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @sub_int_eint_10D(%a0: tensor<1x2x3x4x5x6x7x8x9x10xi3>, %a1: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<1x2x3x4x5x6x7x8x9x10xi3>, tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
  return %1: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
}

// Broadcasting with tensor with dimensions equals to one
// CHECK: func @sub_int_eint_broadcast_1(%[[a0:.*]]: tensor<3x4x1xi3>, %[[a1:.*]]: tensor<1x4x5x!HLFHE.eint<2>>) -> tensor<3x4x5x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.sub_int_eint"(%[[a0]], %[[a1]]) : (tensor<3x4x1xi3>, tensor<1x4x5x!HLFHE.eint<2>>) -> tensor<3x4x5x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<3x4x5x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @sub_int_eint_broadcast_1(%a0: tensor<3x4x1xi3>, %a1: tensor<1x4x5x!HLFHE.eint<2>>) -> tensor<3x4x5x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x4x1xi3>, tensor<1x4x5x!HLFHE.eint<2>>) -> tensor<3x4x5x!HLFHE.eint<2>>
  return %1: tensor<3x4x5x!HLFHE.eint<2>>
}

// Broadcasting with a tensor less dimensions of another
// CHECK: func @sub_int_eint_broadcast_2(%[[a0:.*]]: tensor<3x4xi3>, %[[a1:.*]]: tensor<4x!HLFHE.eint<2>>) -> tensor<3x4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.sub_int_eint"(%[[a0]], %[[a1]]) : (tensor<3x4xi3>, tensor<4x!HLFHE.eint<2>>) -> tensor<3x4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<3x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @sub_int_eint_broadcast_2(%a0: tensor<3x4xi3>, %a1: tensor<4x!HLFHE.eint<2>>) -> tensor<3x4x!HLFHE.eint<2>> {
 %1 ="HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x4xi3>, tensor<4x!HLFHE.eint<2>>) -> tensor<3x4x!HLFHE.eint<2>>
 return %1: tensor<3x4x!HLFHE.eint<2>>
}


/////////////////////////////////////////////////
// HLFHELinalg.neg_eint
/////////////////////////////////////////////////

// 1D tensor
// CHECK: func @neg_eint_1D(%[[a0:.*]]: tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>> {
// CHECK-NEXT: %[[V0:.*]] = "HLFHELinalg.neg_eint"(%[[a0]]) : (tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: return %[[V0]] : tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @neg_eint_1D(%a0: tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.neg_eint"(%a0) : (tensor<4x!HLFHE.eint<2>>) -> tensor<4x!HLFHE.eint<2>>
  return %1: tensor<4x!HLFHE.eint<2>>
}

// 1D tensor
// CHECK: func @neg_eint_2D(%[[a0:.*]]: tensor<4x4x!HLFHE.eint<2>>) -> tensor<4x4x!HLFHE.eint<2>> {
// CHECK-NEXT: %[[V0:.*]] = "HLFHELinalg.neg_eint"(%[[a0]]) : (tensor<4x4x!HLFHE.eint<2>>) -> tensor<4x4x!HLFHE.eint<2>>
// CHECK-NEXT: return %[[V0]] : tensor<4x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @neg_eint_2D(%a0: tensor<4x4x!HLFHE.eint<2>>) -> tensor<4x4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.neg_eint"(%a0) : (tensor<4x4x!HLFHE.eint<2>>) -> tensor<4x4x!HLFHE.eint<2>>
  return %1: tensor<4x4x!HLFHE.eint<2>>
}

// 10D tensor
// CHECK: func @neg_eint_10D(%[[a0:.*]]: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
// CHECK-NEXT: %[[V0:.*]] = "HLFHELinalg.neg_eint"(%[[a0]]) : (tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT: return %[[V0]] : tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @neg_eint_10D(%a0: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.neg_eint"(%a0) : (tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
  return %1: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
}


/////////////////////////////////////////////////
// HLFHELinalg.mul_eint_int
/////////////////////////////////////////////////

// 1D tensor
// CHECK: func @mul_eint_int_1D(%[[a0:.*]]: tensor<4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<4xi3>) -> tensor<4x!HLFHE.eint<2>> {
// CHECK-NEXT: %[[V0:.*]] = "HLFHELinalg.mul_eint_int"(%[[a0]], %[[a1]]) : (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: return %[[V0]] : tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @mul_eint_int_1D(%a0: tensor<4x!HLFHE.eint<2>>, %a1: tensor<4xi3>) -> tensor<4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> tensor<4x!HLFHE.eint<2>>
  return %1: tensor<4x!HLFHE.eint<2>>
}

// 2D tensor
// CHECK: func @mul_eint_int_2D(%[[a0:.*]]: tensor<2x4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<2x4xi3>) -> tensor<2x4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.mul_eint_int"(%[[a0]], %[[a1]]) : (tensor<2x4x!HLFHE.eint<2>>, tensor<2x4xi3>) -> tensor<2x4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<2x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @mul_eint_int_2D(%a0: tensor<2x4x!HLFHE.eint<2>>, %a1: tensor<2x4xi3>) -> tensor<2x4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<2x4x!HLFHE.eint<2>>, tensor<2x4xi3>) -> tensor<2x4x!HLFHE.eint<2>>
  return %1: tensor<2x4x!HLFHE.eint<2>>
}

// 10D tensor
// CHECK: func @mul_eint_int_10D(%[[a0:.*]]: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<1x2x3x4x5x6x7x8x9x10xi3>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.mul_eint_int"(%[[a0]], %[[a1]]) : (tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, tensor<1x2x3x4x5x6x7x8x9x10xi3>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @mul_eint_int_10D(%a0: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, %a1: tensor<1x2x3x4x5x6x7x8x9x10xi3>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>, tensor<1x2x3x4x5x6x7x8x9x10xi3>) -> tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
  return %1: tensor<1x2x3x4x5x6x7x8x9x10x!HLFHE.eint<2>>
}

// Broadcasting with tensor with dimensions equals to one
// CHECK: func @mul_eint_int_broadcast_1(%[[a0:.*]]: tensor<1x4x5x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<3x4x1xi3>) -> tensor<3x4x5x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.mul_eint_int"(%[[a0]], %[[a1]]) : (tensor<1x4x5x!HLFHE.eint<2>>, tensor<3x4x1xi3>) -> tensor<3x4x5x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<3x4x5x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @mul_eint_int_broadcast_1(%a0: tensor<1x4x5x!HLFHE.eint<2>>, %a1: tensor<3x4x1xi3>) -> tensor<3x4x5x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<1x4x5x!HLFHE.eint<2>>, tensor<3x4x1xi3>) -> tensor<3x4x5x!HLFHE.eint<2>>
  return %1: tensor<3x4x5x!HLFHE.eint<2>>
}

// Broadcasting with a tensor less dimensions of another
// CHECK: func @mul_eint_int_broadcast_2(%[[a0:.*]]: tensor<4x!HLFHE.eint<2>>, %[[a1:.*]]: tensor<3x4xi3>) -> tensor<3x4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[V0:.*]] = "HLFHELinalg.mul_eint_int"(%[[a0]], %[[a1]]) : (tensor<4x!HLFHE.eint<2>>, tensor<3x4xi3>) -> tensor<3x4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[V0]] : tensor<3x4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @mul_eint_int_broadcast_2(%a0: tensor<4x!HLFHE.eint<2>>, %a1: tensor<3x4xi3>) -> tensor<3x4x!HLFHE.eint<2>> {
 %1 ="HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<2>>, tensor<3x4xi3>) -> tensor<3x4x!HLFHE.eint<2>>
 return %1: tensor<3x4x!HLFHE.eint<2>>
}

/////////////////////////////////////////////////
// HLFHELinalg.apply_lookup_table
/////////////////////////////////////////////////

// CHECK-LABEL: func @apply_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>>
func @apply_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHELinalg.apply_lookup_table"(%arg0, %arg1) : (tensor<2x3x4x!HLFHE.eint<2>>, tensor<4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>>
  // CHECK-NEXT: return %[[V1]] : tensor<2x3x4x!HLFHE.eint<2>>

  %1 = "HLFHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!HLFHE.eint<2>>, tensor<4xi64>) -> (tensor<2x3x4x!HLFHE.eint<2>>)
  return %1: tensor<2x3x4x!HLFHE.eint<2>>
}

/////////////////////////////////////////////////
// HLFHELinalg.apply_multi_lookup_table
/////////////////////////////////////////////////

// CHECK-LABEL: func @apply_multi_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<2x3x4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>>
func @apply_multi_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<2x3x4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHELinalg.apply_multi_lookup_table"(%arg0, %arg1) : (tensor<2x3x4x!HLFHE.eint<2>>, tensor<2x3x4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>>
  // CHECK-NEXT: return %[[V1]] : tensor<2x3x4x!HLFHE.eint<2>>

  %1 = "HLFHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!HLFHE.eint<2>>, tensor<2x3x4xi64>) -> (tensor<2x3x4x!HLFHE.eint<2>>)
  return %1: tensor<2x3x4x!HLFHE.eint<2>>
}

// CHECK-LABEL: func @apply_multi_lookup_table_broadcast(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<2x4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>>
func @apply_multi_lookup_table_broadcast(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<2x4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHELinalg.apply_multi_lookup_table"(%arg0, %arg1) : (tensor<2x3x4x!HLFHE.eint<2>>, tensor<2x4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>>
  // CHECK-NEXT: return %[[V1]] : tensor<2x3x4x!HLFHE.eint<2>>

  %1 = "HLFHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!HLFHE.eint<2>>, tensor<2x4xi64>) -> (tensor<2x3x4x!HLFHE.eint<2>>)
  return %1: tensor<2x3x4x!HLFHE.eint<2>>
}

/////////////////////////////////////////////////
// HLFHELinalg.dot_eint_int
/////////////////////////////////////////////////

// CHECK-LABEL: func @dot_eint_int(%arg0: tensor<2x!HLFHE.eint<2>>, %arg1: tensor<2xi3>) -> !HLFHE.eint<2>
func @dot_eint_int(%arg0: tensor<2x!HLFHE.eint<2>>,
                   %arg1: tensor<2xi3>) -> !HLFHE.eint<2>
{
  // CHECK-NEXT: %[[RET:.*]] = "HLFHELinalg.dot_eint_int"(%arg0, %arg1) : (tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) -> !HLFHE.eint<2>
  %ret = "HLFHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) -> !HLFHE.eint<2>

  //CHECK-NEXT: return %[[RET]] : !HLFHE.eint<2>
  return %ret : !HLFHE.eint<2>
}

/////////////////////////////////////////////////
// HLFHELinalg.matmul_eint_int
/////////////////////////////////////////////////

// CHECK-LABEL:  @matmul_eint_int(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
func @matmul_eint_int(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1) : (tensor<3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  // CHECK-NEXT: return %[[V1]] : tensor<3x2x!HLFHE.eint<2>>

  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

/////////////////////////////////////////////////
// HLFHELinalg.matmul_int_eint
/////////////////////////////////////////////////

// CHECK-LABEL:  @matmul_int_eint(%arg0: tensor<3x4xi3>, %arg1: tensor<4x2x!HLFHE.eint<2>>) -> tensor<3x2x!HLFHE.eint<2>>
func @matmul_int_eint(%arg0: tensor<3x4xi3>, %arg1: tensor<4x2x!HLFHE.eint<2>>) -> tensor<3x2x!HLFHE.eint<2>> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHELinalg.matmul_int_eint"(%arg0, %arg1) : (tensor<3x4xi3>, tensor<4x2x!HLFHE.eint<2>>) -> tensor<3x2x!HLFHE.eint<2>>
  // CHECK-NEXT: return %[[V1]] : tensor<3x2x!HLFHE.eint<2>>

  %1 = "HLFHELinalg.matmul_int_eint"(%arg0, %arg1): (tensor<3x4xi3>, tensor<4x2x!HLFHE.eint<2>>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

/////////////////////////////////////////////////
// HLFHELinalg.zero
/////////////////////////////////////////////////

// CHECK: func @zero_1D() -> tensor<4x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "HLFHELinalg.zero"() : () -> tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @zero_1D() -> tensor<4x!HLFHE.eint<2>> {
  %0 = "HLFHELinalg.zero"() : () -> tensor<4x!HLFHE.eint<2>>
  return %0 : tensor<4x!HLFHE.eint<2>>
}

// CHECK: func @zero_2D() -> tensor<4x9x!HLFHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "HLFHELinalg.zero"() : () -> tensor<4x9x!HLFHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x9x!HLFHE.eint<2>>
// CHECK-NEXT: }
func @zero_2D() -> tensor<4x9x!HLFHE.eint<2>> {
  %0 = "HLFHELinalg.zero"() : () -> tensor<4x9x!HLFHE.eint<2>>
  return %0 : tensor<4x9x!HLFHE.eint<2>>
}
