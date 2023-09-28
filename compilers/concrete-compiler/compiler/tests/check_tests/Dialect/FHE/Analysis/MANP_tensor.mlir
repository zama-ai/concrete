// RUN: concretecompiler --passes MANP --passes ConcreteOptimizer --action=dump-fhe-no-linalg --split-input-file %s 2>&1 | FileCheck %s

func.func @tensor_from_elements_1(%a: !FHE.eint<2>, %b: !FHE.eint<2>, %c: !FHE.eint<2>, %d: !FHE.eint<2>) -> tensor<4x!FHE.eint<2>>
{
  // The MANP value is 1 as all operands are function arguments
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = tensor.from_elements %a, %b, %c, %d : tensor<4x!FHE.eint<2>>

  return %0 : tensor<4x!FHE.eint<2>>
}

// -----

func.func @tensor_from_elements_2(%a: !FHE.eint<2>, %b: !FHE.eint<2>, %c: !FHE.eint<2>, %d: !FHE.eint<2>) -> tensor<4x!FHE.eint<2>>
{
  %cst = arith.constant 3 : i3

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHE.mul_eint_int"(%a, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  // The MANP value is 3, i.e. the max of all of its operands
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %1 = tensor.from_elements %0, %b, %c, %d : tensor<4x!FHE.eint<2>>

  return %1 : tensor<4x!FHE.eint<2>>
}

// -----

func.func @tensor_extract_1(%t: tensor<4x!FHE.eint<2>>) -> !FHE.eint<2>
{
  %cst = arith.constant 1 : index

  // The MANP value is 1 as the tensor operand is a function argument
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = tensor.extract %t[%cst] : tensor<4x!FHE.eint<2>>

  return %0 : !FHE.eint<2>
}

// -----

func.func @tensor_extract_2(%a: tensor<4x!FHE.eint<2>>) -> !FHE.eint<2>
{
  %c1 = arith.constant 1 : index
  %c3 = arith.constant dense<3> : tensor<4xi3>
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%a, %c3) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %2 = tensor.extract %0[%c1] : tensor<4x!FHE.eint<2>>

  return %2 : !FHE.eint<2>
}

// -----

func.func @tensor_extract_slice_1(%t: tensor<2x10x!FHE.eint<2>>) -> tensor<1x5x!FHE.eint<2>>
{
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = tensor.extract_slice %t[1, 5] [1, 5] [1, 1] : tensor<2x10x!FHE.eint<2>> to tensor<1x5x!FHE.eint<2>>
  
  return %0 : tensor<1x5x!FHE.eint<2>>
}

// -----

func.func @tensor_extract_slice_2(%a: tensor<4x!FHE.eint<2>>) -> tensor<2x!FHE.eint<2>>
{
  %c3 = arith.constant dense <3> : tensor<4xi3>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%a, %c3) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %2 = tensor.extract_slice %0[2] [2] [1] : tensor<4x!FHE.eint<2>> to tensor<2x!FHE.eint<2>>
  
  return %2 : tensor<2x!FHE.eint<2>>
}

// -----

func.func @tensor_insert_slice_1(%t0: tensor<2x10x!FHE.eint<2>>, %t1: tensor<2x2x!FHE.eint<2>>) -> tensor<2x10x!FHE.eint<2>>
{
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = tensor.insert_slice %t1 into %t0[0, 5] [2, 2] [1, 1] : tensor<2x2x!FHE.eint<2>> into tensor<2x10x!FHE.eint<2>>
  
  return %0 : tensor<2x10x!FHE.eint<2>>
}

// -----

func.func @tensor_collapse_shape_1(%a: tensor<2x2x4x!FHE.eint<6>>) -> tensor<2x8x!FHE.eint<6>> {
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = tensor.collapse_shape %a [[0],[1,2]]  : tensor<2x2x4x!FHE.eint<6>> into tensor<2x8x!FHE.eint<6>>
  return %0 : tensor<2x8x!FHE.eint<6>>
}

// -----

func.func @tensor_collapse_shape_2(%a: tensor<2x2x4x!FHE.eint<2>>, %b: tensor<2x2x4xi3>) -> tensor<2x8x!FHE.eint<2>>
{
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%a, %b) : (tensor<2x2x4x!FHE.eint<2>>, tensor<2x2x4xi3>) -> tensor<2x2x4x!FHE.eint<2>>
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %1 = tensor.collapse_shape %0 [[0],[1,2]]  : tensor<2x2x4x!FHE.eint<2>> into tensor<2x8x!FHE.eint<2>>
  return %1 : tensor<2x8x!FHE.eint<2>>
}

// -----

func.func @tensor_expand_shape_1(%a: tensor<2x8x!FHE.eint<6>>) -> tensor<2x2x4x!FHE.eint<6>> {
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = tensor.expand_shape %a [[0],[1,2]]  : tensor<2x8x!FHE.eint<6>> into tensor<2x2x4x!FHE.eint<6>>
  return %0 : tensor<2x2x4x!FHE.eint<6>>
}

// -----

func.func @tensor_expand_shape_2(%a: tensor<2x8x!FHE.eint<2>>, %b: tensor<2x8xi3>) -> tensor<2x2x4x!FHE.eint<2>>
{
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%a, %b) : (tensor<2x8x!FHE.eint<2>>, tensor<2x8xi3>) -> tensor<2x8x!FHE.eint<2>>
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %1 = tensor.expand_shape %0 [[0],[1,2]]  : tensor<2x8x!FHE.eint<2>> into tensor<2x2x4x!FHE.eint<2>>
  return %1 : tensor<2x2x4x!FHE.eint<2>>
}
