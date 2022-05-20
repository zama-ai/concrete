// RUN: concretecompiler --passes MANP --action=dump-fhe --split-input-file %s 2>&1 | FileCheck %s

func @tensor_from_elements_1(%a: !FHE.eint<2>, %b: !FHE.eint<2>, %c: !FHE.eint<2>, %d: !FHE.eint<2>) -> tensor<4x!FHE.eint<2>>
{
  // The MANP value is 1 as all operands are function arguments
  // CHECK: %[[ret:.*]] = tensor.from_elements %[[a:.*]], %[[b:.*]], %[[c:.*]], %[[d:.*]] {MANP = 1 : ui{{[[0-9]+}}} : tensor<4x!FHE.eint<2>>
  %0 = tensor.from_elements %a, %b, %c, %d : tensor<4x!FHE.eint<2>>

  return %0 : tensor<4x!FHE.eint<2>>
}

// -----

func @tensor_from_elements_2(%a: !FHE.eint<2>, %b: !FHE.eint<2>, %c: !FHE.eint<2>, %d: !FHE.eint<2>) -> tensor<4x!FHE.eint<2>>
{
  %cst = arith.constant 3 : i3

  // CHECK: %[[V0:.*]] = "FHE.add_eint_int"(%[[a:.*]], %[[cst:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %0 = "FHE.add_eint_int"(%a, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  // The MANP value is 4, i.e. the max of all of its operands
  // CHECK: %[[V1:.*]] = tensor.from_elements %[[V0]], %[[b:.*]], %[[c:.*]], %[[d:.*]] {MANP = 4 : ui{{[[0-9]+}}} : tensor<4x!FHE.eint<2>>
  %1 = tensor.from_elements %0, %b, %c, %d : tensor<4x!FHE.eint<2>>

  return %1 : tensor<4x!FHE.eint<2>>
}

// -----

func @tensor_extract_1(%t: tensor<4x!FHE.eint<2>>) -> !FHE.eint<2>
{
  %cst = arith.constant 1 : index

  // The MANP value is 1 as the tensor operand is a function argument
  // CHECK: %[[ret:.*]] = tensor.extract %[[t:.*]][%[[c1:.*]]] {MANP = 1 : ui{{[[0-9]+}}} : tensor<4x!FHE.eint<2>>
  %0 = tensor.extract %t[%cst] : tensor<4x!FHE.eint<2>>

  return %0 : !FHE.eint<2>
}

// -----

func @tensor_extract_2(%a: tensor<4x!FHE.eint<2>>) -> !FHE.eint<2>
{
  %c1 = arith.constant 1 : index
  %c3 = arith.constant dense<3> : tensor<4xi3>
  // CHECK: %[[V0:.*]] = "FHELinalg.add_eint_int"(%[[a:.*]], %[[c1:.*]]) {MANP = 4 : ui{{[0-9]+}}}
  %0 = "FHELinalg.add_eint_int"(%a, %c3) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>
  // CHECK: %[[ret:.*]] = tensor.extract %[[V0]][%[[c3:.*]]] {MANP = 4 : ui{{[[0-9]+}}} : tensor<4x!FHE.eint<2>>
  %2 = tensor.extract %0[%c1] : tensor<4x!FHE.eint<2>>

  return %2 : !FHE.eint<2>
}

// -----

func @tensor_extract_slice_1(%t: tensor<2x10x!FHE.eint<2>>) -> tensor<1x5x!FHE.eint<2>>
{
  // CHECK: %[[V0:.*]] = tensor.extract_slice %[[t:.*]][1, 5] [1, 5] [1, 1] {MANP = 1 : ui{{[[0-9]+}}} : tensor<2x10x!FHE.eint<2>> to tensor<1x5x!FHE.eint<2>>
  %0 = tensor.extract_slice %t[1, 5] [1, 5] [1, 1] : tensor<2x10x!FHE.eint<2>> to tensor<1x5x!FHE.eint<2>>
  
  return %0 : tensor<1x5x!FHE.eint<2>>
}

// -----

func @tensor_extract_slice_2(%a: tensor<4x!FHE.eint<2>>) -> tensor<2x!FHE.eint<2>>
{
  %c3 = arith.constant dense <3> : tensor<4xi3>

  // CHECK: %[[V0:.*]] = "FHELinalg.add_eint_int"(%[[a:.*]], %[[c1:.*]]) {MANP = 4 : ui{{[0-9]+}}}
  %0 = "FHELinalg.add_eint_int"(%a, %c3) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>

  // CHECK: tensor.extract_slice %[[V0]][2] [2] [1] {MANP = 4 : ui{{[0-9]+}}} : tensor<4x!FHE.eint<2>> to tensor<2x!FHE.eint<2>>
  %2 = tensor.extract_slice %0[2] [2] [1] : tensor<4x!FHE.eint<2>> to tensor<2x!FHE.eint<2>>
  
  return %2 : tensor<2x!FHE.eint<2>>
}

// -----

func @tensor_insert_slice_1(%t0: tensor<2x10x!FHE.eint<2>>, %t1: tensor<2x2x!FHE.eint<2>>) -> tensor<2x10x!FHE.eint<2>>
{
  // %[[V0:.*]] = tensor.insert_slice %[[t1:.*]] into %[[t0:.*]][0, 5] [2, 2] [1, 1] {MANP = 1 : ui{{[[0-9]+}}} : tensor<2x2x!FHE.eint<2>> into tensor<2x10x!FHE.eint<2>>
  %0 = tensor.insert_slice %t1 into %t0[0, 5] [2, 2] [1, 1] : tensor<2x2x!FHE.eint<2>> into tensor<2x10x!FHE.eint<2>>
  
  return %0 : tensor<2x10x!FHE.eint<2>>
}

// -----

func @tensor_collapse_shape_1(%a: tensor<2x2x4x!FHE.eint<6>>) -> tensor<2x8x!FHE.eint<6>> {
  // CHECK: linalg.tensor_collapse_shape %[[A:.*]] [[X:.*]] {MANP = 1 : ui{{[0-9]+}}}
  %0 = linalg.tensor_collapse_shape %a [[0],[1,2]]  : tensor<2x2x4x!FHE.eint<6>> into tensor<2x8x!FHE.eint<6>>
  return %0 : tensor<2x8x!FHE.eint<6>>
}

// -----

func @tensor_collapse_shape_2(%a: tensor<2x2x4x!FHE.eint<2>>, %b: tensor<2x2x4xi3>) -> tensor<2x8x!FHE.eint<2>>
{
  // CHECK: "FHELinalg.add_eint_int"(%[[A:.*]], %[[B:.*]]) {MANP = 4 : ui{{[0-9]+}}}
  %0 = "FHELinalg.add_eint_int"(%a, %b) : (tensor<2x2x4x!FHE.eint<2>>, tensor<2x2x4xi3>) -> tensor<2x2x4x!FHE.eint<2>>
  // CHECK-NEXT: linalg.tensor_collapse_shape %[[A:.*]] [[X:.*]] {MANP = 4 : ui{{[0-9]+}}}
  %1 = linalg.tensor_collapse_shape %0 [[0],[1,2]]  : tensor<2x2x4x!FHE.eint<2>> into tensor<2x8x!FHE.eint<2>>
  return %1 : tensor<2x8x!FHE.eint<2>>
}

// -----

func @tensor_expand_shape_1(%a: tensor<2x8x!FHE.eint<6>>) -> tensor<2x2x4x!FHE.eint<6>> {
  // CHECK: linalg.tensor_expand_shape %[[A:.*]] [[X:.*]] {MANP = 1 : ui{{[0-9]+}}}
  %0 = linalg.tensor_expand_shape %a [[0],[1,2]]  : tensor<2x8x!FHE.eint<6>> into tensor<2x2x4x!FHE.eint<6>>
  return %0 : tensor<2x2x4x!FHE.eint<6>>
}

// -----

func @tensor_expand_shape_2(%a: tensor<2x8x!FHE.eint<2>>, %b: tensor<2x8xi3>) -> tensor<2x2x4x!FHE.eint<2>>
{
  // CHECK: "FHELinalg.add_eint_int"(%[[A:.*]], %[[B:.*]]) {MANP = 4 : ui{{[0-9]+}}}
  %0 = "FHELinalg.add_eint_int"(%a, %b) : (tensor<2x8x!FHE.eint<2>>, tensor<2x8xi3>) -> tensor<2x8x!FHE.eint<2>>
  // CHECK-NEXT: linalg.tensor_expand_shape %[[A:.*]] [[X:.*]] {MANP = 4 : ui{{[0-9]+}}}
  %1 = linalg.tensor_expand_shape %0 [[0],[1,2]]  : tensor<2x8x!FHE.eint<2>> into tensor<2x2x4x!FHE.eint<2>>
  return %1 : tensor<2x2x4x!FHE.eint<2>>
}