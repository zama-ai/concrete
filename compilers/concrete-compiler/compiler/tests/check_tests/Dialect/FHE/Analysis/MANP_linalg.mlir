// RUN: concretecompiler --passes canonicalize --passes MANP --passes ConcreteOptimizer --optimizer-strategy=dag-mono --action=dump-fhe-no-linalg --split-input-file %s 2>&1 | FileCheck %s

func.func @single_cst_add_eint_int(%t: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.add_eint_int"(%t, %cst) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_cst_add_eint_int_from_cst_elements(%t: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst1 = arith.constant 1 : i3
  %cst = tensor.from_elements %cst1, %cst1, %cst1, %cst1, %cst1, %cst1, %cst1, %cst1: tensor<8xi3>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.add_eint_int"(%t, %cst) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----
func.func @single_dyn_add_eint_int(%e: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.add_eint_int"(%e, %i) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_add_eint(%e0: tensor<8x!FHE.eint<2>>, %e1: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %0 = "FHELinalg.add_eint"(%e0, %e1) : (tensor<8x!FHE.eint<2>>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_cst_sub_int_eint(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.sub_int_eint"(%cst, %e) : (tensor<8xi3>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_cst_sub_int_eint_from_cst_elements(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst1 = arith.constant 1 : i3
  %cst = tensor.from_elements %cst1, %cst1, %cst1, %cst1, %cst1, %cst1, %cst1, %cst1: tensor<8xi3>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.sub_int_eint"(%cst, %e) : (tensor<8xi3>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_cst_sub_eint_int(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.sub_eint_int"(%e, %cst) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_cst_sub_eint_int_from_cst_elements(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst1 = arith.constant 1 : i3
  %cst = tensor.from_elements %cst1, %cst1, %cst1, %cst1, %cst1, %cst1, %cst1, %cst1: tensor<8xi3>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.sub_eint_int"(%e, %cst) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_sub_eint(%e0: tensor<8x!FHE.eint<2>>, %e1: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %0 = "FHELinalg.sub_eint"(%e0, %e1) : (tensor<8x!FHE.eint<2>>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_neg_eint(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.neg_eint"(%e) : (tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_dyn_sub_int_eint(%e: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
{
  // sqrt(1 + (2^2-1)^2) = 3.16
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.sub_int_eint"(%i, %e) : (tensor<8xi3>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_cst_mul_eint_int(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%e, %cst) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_cst_mul_eint_int_from_cst_elements(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst1 = arith.constant 2 : i3
  %cst = tensor.from_elements %cst1, %cst1, %cst1, %cst1, %cst1, %cst1, %cst1, %cst1: tensor<8xi3>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%e, %cst) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_dyn_mul_eint_int(%e: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
{
  // sqrt(1 * (2^2-1)^2) = 3.16
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%e, %i) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @chain_add_eint_int(%e: tensor<8x!FHE.eint<3>>) -> tensor<8x!FHE.eint<3>>
{
  %cst0 = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi4>
  %cst1 = arith.constant dense<[0, 7, 2, 5, 6, 2, 1, 7]> : tensor<8xi4>
  %cst2 = arith.constant dense<[0, 1, 2, 0, 1, 2, 0, 1]> : tensor<8xi4>
  %cst3 = arith.constant dense<[0, 1, 1, 0, 0, 1, 0, 1]> : tensor<8xi4>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.add_eint_int"(%e, %cst0) : (tensor<8x!FHE.eint<3>>, tensor<8xi4>) -> tensor<8x!FHE.eint<3>>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %1 = "FHELinalg.add_eint_int"(%0, %cst1) : (tensor<8x!FHE.eint<3>>, tensor<8xi4>) -> tensor<8x!FHE.eint<3>>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %2 = "FHELinalg.add_eint_int"(%1, %cst2) : (tensor<8x!FHE.eint<3>>, tensor<8xi4>) -> tensor<8x!FHE.eint<3>>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %3 = "FHELinalg.add_eint_int"(%2, %cst3) : (tensor<8x!FHE.eint<3>>, tensor<8xi4>) -> tensor<8x!FHE.eint<3>>
  return %3 : tensor<8x!FHE.eint<3>>
}

// -----

func.func @chain_add_eint_int_neg_eint(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst0 = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHELinalg.add_eint_int"(%e, %cst0) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %1 = "FHELinalg.neg_eint"(%0) : (tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
  return %1 : tensor<8x!FHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.apply_multi_lookup_table
/////////////////////////////////////////////////

func.func @apply_lookup_table(%t: tensor<3x3x!FHE.eint<2>>) -> tensor<3x3x!FHE.eint<3>> {
  %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %res = "FHELinalg.apply_lookup_table"(%t, %lut) : (tensor<3x3x!FHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!FHE.eint<3>>
  return %res : tensor<3x3x!FHE.eint<3>>
}

// -----

func.func @apply_lookup_table_after_op(%t: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!FHE.eint<3>> {
  %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%t, %i) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %res = "FHELinalg.apply_lookup_table"(%0, %lut) : (tensor<8x!FHE.eint<2>>, tensor<4xi64>) -> tensor<8x!FHE.eint<3>>
  return %res : tensor<8x!FHE.eint<3>>
}

// -----


func.func @apply_multi_lookup_table(%t: tensor<3x3x!FHE.eint<2>>, %luts: tensor<3x3x4xi64>) -> tensor<3x3x!FHE.eint<3>> {
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %res = "FHELinalg.apply_multi_lookup_table"(%t, %luts) : (tensor<3x3x!FHE.eint<2>>, tensor<3x3x4xi64>) -> tensor<3x3x!FHE.eint<3>>
  return %res : tensor<3x3x!FHE.eint<3>>
}

// -----

func.func @apply_multi_lookup_table_after_op(%t: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>, %luts: tensor<8x4xi64>) -> tensor<8x!FHE.eint<3>> {
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%t, %i) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %res = "FHELinalg.apply_multi_lookup_table"(%0, %luts) : (tensor<8x!FHE.eint<2>>, tensor<8x4xi64>) -> tensor<8x!FHE.eint<3>>
  return %res : tensor<8x!FHE.eint<3>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.dot_eint_int
/////////////////////////////////////////////////

func.func @single_cst_dot(%t: tensor<4x!FHE.eint<2>>) -> !FHE.eint<2>
{
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi3>
  // sqrt(1^2*1 + 2^2*1 + 3^2*1 + 4^2*1) = 5.477225575
  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %0 = "FHELinalg.dot_eint_int"(%t, %cst) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>
  return %0 : !FHE.eint<2>
}


// -----

func.func @single_dyn_dot(%t: tensor<4x!FHE.eint<2>>, %dyn: tensor<4xi3>) -> !FHE.eint<2>
{
  // sqrt(1^2*(2^2-1)^2*4) = 6
  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %0 = "FHELinalg.dot_eint_int"(%t, %dyn) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_dot_after_op(%t: tensor<4x!FHE.eint<2>>, %i: tensor<4xi3>) -> !FHE.eint<2>
{
  // sqrt((2^2-1)^2*1) = sqrt(9) = 3
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%t, %i) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>

  %cst = arith.constant dense<[1, 2, 3, -1]> : tensor<4xi3>
  // sqrt(1^2*9 + 2^2*9 + 3^2*9 + 1^2*9) = sqrt(135) = 12
  // CHECK: MANP = 12 : ui{{[0-9]+}}
  %1 = "FHELinalg.dot_eint_int"(%0, %cst) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>

  return %1 : !FHE.eint<2>
}

// -----

func.func @single_dyn_dot_after_op(%t: tensor<4x!FHE.eint<2>>, %i: tensor<4xi3>) -> !FHE.eint<2>
{
  // sqrt((2^2-1)^2*1) = sqrt(9) = 3
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.mul_eint_int"(%t, %i) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>

  // sqrt(3^2*(2^2-1)^2*4) = sqrt(324) = 18
  // CHECK: MANP = 18 : ui{{[0-9]+}}
  %1 = "FHELinalg.dot_eint_int"(%0, %i) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>

  return %1 : !FHE.eint<2>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.matmul_ent_int
/////////////////////////////////////////////////

func.func @matmul_eint_int_dyn_p_1(%arg0: tensor<3x1x!FHE.eint<2>>, %arg1: tensor<1x2xi3>) -> tensor<3x2x!FHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 0
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^2-1)^2 = 9
  // manp(add_eint(mul, acc)) = 9
  // ceil(sqrt(9)) = 3
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x1x!FHE.eint<2>>, tensor<1x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %0 : tensor<3x2x!FHE.eint<2>>
}

// -----

func.func @matmul_eint_int_dyn_p_2(%arg0: tensor<3x2x!FHE.eint<2>>, %arg1: tensor<2x2xi3>) -> tensor<3x2x!FHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 0
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^2-1)^2 = 9
  // manp(add_eint(mul, acc)) = 9
  // p = 1
  // manp(mul_eint_int(eint<2>, i3) = 1 * (2^2-1)^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 9 = 18
  // ceil(sqrt(18)) = 5
  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x2x!FHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func.func @matmul_eint_int_cst_p_1(%arg0: tensor<3x1x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  %0 = arith.constant dense<[[3, 1]]> : tensor<1x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max cst is used for n = 0
  // acc = manp(0) = 0
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9
  // ceil(sqrt(10)) = 3
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x1x!FHE.eint<2>>, tensor<1x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func.func @matmul_eint_int_cst_p_2_n_0(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  %0 = arith.constant dense<[[4, 1],[3, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,3] are used for n = 0
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 1 = 10
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 16
  // manp(add_eint(mul, acc)) = 16 + 9 = 25
  // ceil(sqrt(25)) = 5
  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x2x!FHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func.func @matmul_eint_int_cst_p_2_n_1(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  %0 = arith.constant dense<[[1, 4],[3, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,1] are used for n = 1
  // p = 0
  // acc = manp(0) = 0
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 16
  // manp(add_eint(mul, acc)) = 16
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 1) = 1 * 1^2 = 1
  // manp(add_eint(mul, acc)) = 1 + 16 = 17
  // ceil(sqrt(17)) = 5
  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x2x!FHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func.func @matmul_eint_int_cst(%0: tensor<4x3x!FHE.eint<7>>) -> (tensor<4x!FHE.eint<7>>, tensor<4x2x!FHE.eint<7>>, tensor<5x4x2x!FHE.eint<7>>, tensor<2x5x4x2x!FHE.eint<7>>) {

  // ===============================

  %1 = arith.constant dense<
    // ceil(sqrt(2^2 + 1^2 + 5^2)) = ceil(sqrt(30)) = 6
    [2, 1, 5]
  > : tensor<3xi8>

  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %2 = "FHELinalg.matmul_eint_int"(%0, %1) : (tensor<4x3x!FHE.eint<7>>, tensor<3xi8>) -> tensor<4x!FHE.eint<7>>

  // ===============================

  %3 = arith.constant dense<
    // ceil(sqrt(2^2 + 3^2 + 5^2)) = ceil(sqrt(39)) = 7
    // ceil(sqrt(3^2 + 2^2 + 6^2)) = ceil(sqrt(49)) = 7
    [
      [2, 3],
      [3, 2],
      [5, 6]
    ]
  > : tensor<3x2xi8>

  // CHECK: MANP = 7 : ui{{[0-9]+}}
  %4 = "FHELinalg.matmul_eint_int"(%0, %3) : (tensor<4x3x!FHE.eint<7>>, tensor<3x2xi8>) -> tensor<4x2x!FHE.eint<7>>

  // ===============================

  %5 = arith.constant dense<
    [
      // ceil(sqrt(1^2 + 4^2 + 6^2)) = ceil(sqrt(53)) = 8
      // ceil(sqrt(6^2 + 3^2 + 2^2)) = ceil(sqrt(49)) = 7
      [
        [1, 6],
        [4, 3],
        [6, 2]
      ],

      // ceil(sqrt(5^2 + 3^2 + 5^2)) = ceil(sqrt(59)) = 8
      // ceil(sqrt(3^2 + 2^2 + 6^2)) = ceil(sqrt(49)) = 7
      [
        [5, 3],
        [3, 2],
        [5, 6]
      ],

      // ceil(sqrt(5^2 + 5^2 + 3^2)) = ceil(sqrt(59)) = 8
      // ceil(sqrt(3^2 + 6^2 + 3^2)) = ceil(sqrt(54)) = 8
      [
        [5, 3],
        [5, 6],
        [3, 3]
      ],

      // ceil(sqrt(6^2 + 1^2 + 4^2)) = ceil(sqrt(53)) = 8
      // ceil(sqrt(3^2 + 4^2 + 3^2)) = ceil(sqrt(34)) = 6
      [
        [6, 3],
        [1, 4],
        [4, 3]
      ],

      // ceil(sqrt(1^2 + 6^2 + 6^2)) = ceil(sqrt(73)) = 9
      // ceil(sqrt(2^2 + 1^2 + 5^2)) = ceil(sqrt(30)) = 6
      [
        [1, 2],
        [6, 1],
        [6, 5]
      ]
   ]
  > : tensor<5x3x2xi8>

  // CHECK: MANP = 9 : ui{{[0-9]+}}
  %6 = "FHELinalg.matmul_eint_int"(%0, %5) : (tensor<4x3x!FHE.eint<7>>, tensor<5x3x2xi8>) -> tensor<5x4x2x!FHE.eint<7>>

  // ===============================

  %7 = arith.constant dense<
    [
      [
        // ceil(sqrt(1^2 + 4^2 + 6^2)) = ceil(sqrt(53)) = 8
        // ceil(sqrt(6^2 + 3^2 + 2^2)) = ceil(sqrt(49)) = 7
        [
          [1, 6],
          [4, 3],
          [6, 2]
        ],

        // ceil(sqrt(5^2 + 3^2 + 5^2)) = ceil(sqrt(59)) = 8
        // ceil(sqrt(3^2 + 2^2 + 6^2)) = ceil(sqrt(47)) = 7
        [
          [5, 3],
          [3, 2],
          [5, 6]
        ],

        // ceil(sqrt(5^2 + 5^2 + 3^2)) = ceil(sqrt(59)) = 8
        // ceil(sqrt(3^2 + 6^2 + 3^2)) = ceil(sqrt(54)) = 8
        [
          [5, 3],
          [5, 6],
          [3, 3]
        ],

        // ceil(sqrt(6^2 + 1^2 + 4^2)) = ceil(sqrt(53)) = 8
        // ceil(sqrt(3^2 + 4^2 + 3^2)) = ceil(sqrt(34)) = 6
        [
          [6, 3],
          [1, 4],
          [4, 3]
        ],

        // ceil(sqrt(1^2 + 6^2 + 6^2)) = ceil(sqrt(73)) = 9
        // ceil(sqrt(2^2 + 1^2 + 5^2)) = ceil(sqrt(30)) = 6
        [
          [1, 2],
          [6, 1],
          [6, 5]
        ]
      ],
      [
        // ceil(sqrt(6^2 + 1^2 + 3^2)) = ceil(sqrt(46)) = 7
        // ceil(sqrt(5^2 + 6^2 + 6^2)) = ceil(sqrt(97)) = 10
        [
          [6, 5],
          [1, 6],
          [3, 6]
        ],

        // ceil(sqrt(1^2 + 2^2 + 5^2)) = ceil(sqrt(30)) = 6
        // ceil(sqrt(6^2 + 3^2 + 1^2)) = ceil(sqrt(46)) = 7
        [
          [1, 6],
          [2, 3],
          [5, 1]
        ],

        // ceil(sqrt(4^2 + 3^2 + 6^2)) = ceil(sqrt(61)) = 8
        // ceil(sqrt(1^2 + 5^2 + 2^2)) = ceil(sqrt(30)) = 6
        [
          [4, 1],
          [3, 5],
          [6, 2]
        ],

        // ceil(sqrt(2^2 + 3^2 + 3^2)) = ceil(sqrt(22)) = 5
        // ceil(sqrt(2^2 + 2^2 + 1^2)) = ceil(sqrt(9)) = 3
        [
          [2, 2],
          [3, 2],
          [3, 1]
        ],

        // ceil(sqrt(6^2 + 2^2 + 3^2)) = ceil(sqrt(49)) = 7
        // ceil(sqrt(2^2 + 4^2 + 2^2)) = ceil(sqrt(24)) = 5
        [
          [6, 2],
          [2, 4],
          [3, 2]
        ]
      ]
    ]
  > : tensor<2x5x3x2xi8>

  // CHECK: MANP = 10 : ui{{[0-9]+}}
  %8 = "FHELinalg.matmul_eint_int"(%0, %7) : (tensor<4x3x!FHE.eint<7>>, tensor<2x5x3x2xi8>) -> tensor<2x5x4x2x!FHE.eint<7>>

  // ===============================

  return %2, %4, %6, %8 : tensor<4x!FHE.eint<7>>, tensor<4x2x!FHE.eint<7>>, tensor<5x4x2x!FHE.eint<7>>, tensor<2x5x4x2x!FHE.eint<7>>
}

// -----

func.func @matmul_eint_int_cst_different_operand_manp() -> tensor<4x!FHE.eint<7>> {
  // CHECK: {MANP = 0 : ui{{[0-9]+}}}
  %z = "FHE.zero_tensor"() : () -> tensor<4x3x!FHE.eint<7>>
  %a = arith.constant dense<[[4, 6, 5], [2, 6, 3], [5, 6, 1], [5, 5, 3]]> : tensor<4x3xi8>

  // CHECK: {MANP = 0 : ui{{[0-9]+}}}
  %0 = "FHELinalg.add_eint_int"(%z, %a) : (tensor<4x3x!FHE.eint<7>>, tensor<4x3xi8>) -> tensor<4x3x!FHE.eint<7>>

  // ===============================

  %1 = arith.constant dense<
    // ceil(sqrt(1 * (2^2 + 1^2 + 5^2))) = ceil(sqrt(30)) = 6
    [2, 1, 5]
  > : tensor<3xi8>

  // CHECK: MANP = 0 : ui{{[0-9]+}}
  %2 = "FHELinalg.matmul_eint_int"(%0, %1) : (tensor<4x3x!FHE.eint<7>>, tensor<3xi8>) -> tensor<4x!FHE.eint<7>>

  // ===============================

  return %2 : tensor<4x!FHE.eint<7>>
}

/////////////////////////////////////////////////
// FHELinalg.matmul_int_eint
/////////////////////////////////////////////////

// -----

func.func @matmul_int_eint_dyn_p_1(%arg0: tensor<3x1xi3>, %arg1: tensor<1x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 0
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^2-1)^2 = 9
  // manp(add_eint(mul, acc)) = 9
  // ceil(sqrt(9)) = 3
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_int_eint"(%arg0, %arg1): (tensor<3x1xi3>, tensor<1x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func.func @matmul_int_eint_dyn_p_2(%arg0: tensor<3x2xi3>, %arg1: tensor<2x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 0
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^2-1)^2 = 9
  // manp(add_eint(mul, acc)) = 0 + 9 = 9
  // manp(mul_eint_int(eint<2>, i3) = 1 * (2^2-1)^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 9 = 18
  // ceil(sqrt(18)) = 5
  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_int_eint"(%arg0, %arg1): (tensor<3x2xi3>, tensor<2x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func.func @matmul_int_eint_cst_p_1(%arg0: tensor<1x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>> {
  %0 = arith.constant dense<[[3], [1]]> : tensor<2x1xi3>
  // c(m,n) = a(m,p) * b(p,n) the max cst is used for m = 0
  // acc = manp(0) = 0
  // mul = manp(mul_eint_int(eint<2>, 3) = 1^2 + 3^2 = 10
  // manp(add_eint(mul, acc)) = 10
  // ceil(sqrt(10)) = 4
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_int_eint"(%0, %arg0): (tensor<2x1xi3>, tensor<1x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>>
  return %1 : tensor<2x3x!FHE.eint<2>>
}

// -----

func.func @matmul_int_eint_cst_p_2_n_0(%arg0: tensor<2x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>> {
  %0 = arith.constant dense<[[3, 4],[1, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,3] are used for m = 0
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 1 = 10
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 17
  // manp(add_eint(mul, acc)) = 17 + 9 = 26
  // ceil(sqrt(26)) = 6
  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_int_eint"(%0, %arg0): (tensor<2x2xi3>, tensor<2x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>>
  return %1 : tensor<2x3x!FHE.eint<2>>
}

// -----

func.func @matmul_int_eint_cst_p_2_n_1(%arg0: tensor<2x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>> {
  %0 = arith.constant dense<[[4, 1],[3, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,1] are used for m = 1
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 16
  // manp(add_eint(mul, acc)) = 16 + 1 = 17
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 1) = 1 * 1^2 = 1
  // manp(add_eint(mul, acc)) = 1 + 17 = 18
  // ceil(sqrt(18)) = 5
  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %1 = "FHELinalg.matmul_int_eint"(%0, %arg0): (tensor<2x2xi3>, tensor<2x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>>
  return %1 : tensor<2x3x!FHE.eint<2>>
}

// -----

func.func @matmul_int_eint_cst(%0: tensor<3x2x!FHE.eint<7>>) -> (tensor<2x!FHE.eint<7>>, tensor<2x2x!FHE.eint<7>>, tensor<5x2x2x!FHE.eint<7>>, tensor<2x5x2x2x!FHE.eint<7>>) {

  // ===============================

  %1 = arith.constant dense<
    // ceil(sqrt(2^2 + 1^2 + 5^2)) = ceil(sqrt(30)) = 6
    [2, 1, 5]
  > : tensor<3xi8>

  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %2 = "FHELinalg.matmul_int_eint"(%1, %0) : (tensor<3xi8>, tensor<3x2x!FHE.eint<7>>) -> tensor<2x!FHE.eint<7>>

  // ===============================

  %3 = arith.constant dense<
    // ceil(sqrt(2^2 + 3^2 + 5^2)) = ceil(sqrt(38)) = 7
    // ceil(sqrt(3^2 + 2^2 + 6^2)) = ceil(sqrt(49)) = 7
    [
      [2, 3, 5],
      [3, 2, 6]
    ]
  > : tensor<2x3xi8>

  // CHECK: MANP = 7 : ui{{[0-9]+}}
  %4 = "FHELinalg.matmul_int_eint"(%3, %0) : (tensor<2x3xi8>, tensor<3x2x!FHE.eint<7>>) -> tensor<2x2x!FHE.eint<7>>

  // ===============================

  %5 = arith.constant dense<
    [
      // ceil(sqrt(1^2 + 4^2 + 6^2)) = ceil(sqrt(53)) = 8
      // ceil(sqrt(6^2 + 3^2 + 2^2)) = ceil(sqrt(49)) = 7
      [
        [1, 4, 6],
        [6, 3, 2]
      ],

      // ceil(sqrt(5^2 + 3^2 + 5^2)) = ceil(sqrt(59)) = 8
      // ceil(sqrt(3^2 + 2^2 + 6^2)) = ceil(sqrt(49)) = 7
      [
        [5, 3, 5],
        [3, 2, 6]
      ],

      // ceil(sqrt(5^2 + 5^2 + 3^2)) = ceil(sqrt(59)) = 8
      // ceil(sqrt(3^2 + 6^2 + 3^2)) = ceil(sqrt(54)) = 8
      [
        [5, 5, 3],
        [3, 6, 3]
      ],

      // ceil(sqrt(6^2 + 1^2 + 4^2)) = ceil(sqrt(53)) = 8
      // ceil(sqrt(3^2 + 4^2 + 3^2)) = ceil(sqrt(34)) = 6
      [
        [6, 1, 4],
        [3, 4, 3]
      ],

      // ceil(sqrt(1^2 + 6^2 + 6^2)) = ceil(sqrt(73)) = 9
      // ceil(sqrt(2^2 + 1^2 + 5^2)) = ceil(sqrt(30)) = 6
      [
        [1, 6, 6],
        [2, 1, 5]
      ]
   ]
  > : tensor<5x2x3xi8>

  // CHECK: MANP = 9 : ui{{[0-9]+}}
  %6 = "FHELinalg.matmul_int_eint"(%5, %0) : (tensor<5x2x3xi8>, tensor<3x2x!FHE.eint<7>>) -> tensor<5x2x2x!FHE.eint<7>>

  // ===============================

  %7 = arith.constant dense<
    [
      [
        // ceil(sqrt(1^2 + 4^2 + 6^2)) = ceil(sqrt(53)) = 8
        // ceil(sqrt(6^2 + 3^2 + 2^2)) = ceil(sqrt(49)) = 7
        [
          [1, 4, 6],
          [6, 3, 2]
        ],

        // ceil(sqrt(5^2 + 3^2 + 5^2)) = ceil(sqrt(59)) = 8
        // ceil(sqrt(3^2 + 2^2 + 6^2)) = ceil(sqrt(49)) = 7
        [
          [5, 3, 5],
          [3, 2, 6]
        ],

        // ceil(sqrt(5^2 + 5^2 + 3^2)) = ceil(sqrt(59)) = 8
        // ceil(sqrt(3^2 + 6^2 + 3^2)) = ceil(sqrt(54)) = 8
        [
          [5, 5, 3],
          [3, 6, 3]
        ],

        // ceil(sqrt(6^2 + 1^2 + 4^2)) = ceil(sqrt(53)) = 8
        // ceil(sqrt(3^2 + 4^2 + 3^2)) = ceil(sqrt(34)) = 6
        [
          [6, 1, 4],
          [3, 4, 3]
        ],

        // ceil(sqrt(1^2 + 6^2 + 6^2)) = ceil(sqrt(73)) = 9
        // ceil(sqrt(2^2 + 1^2 + 5^2)) = ceil(sqrt(30)) = 6
        [
          [1, 6, 6],
          [2, 1, 5]
        ]
      ],
      [
        // ceil(sqrt(6^2 + 1^2 + 3^2)) = ceil(sqrt(46)) = 7
        // ceil(sqrt(5^2 + 6^2 + 6^2)) = ceil(sqrt(97)) = 10
        [
          [6, 1, 3],
          [5, 6, 6]
        ],

        // ceil(sqrt(1^2 + 2^2 + 5^2)) = ceil(sqrt(30)) = 6
        // ceil(sqrt(6^2 + 3^2 + 1^2)) = ceil(sqrt(46)) = 7
        [
          [1, 2, 5],
          [6, 3, 1]
        ],

        // ceil(sqrt(4^2 + 3^2 + 6^2)) = ceil(sqrt(61)) = 8
        // ceil(sqrt(1^2 + 5^2 + 2^2)) = ceil(sqrt(30)) = 6
        [
          [4, 3, 6],
          [1, 5, 2]
        ],

        // ceil(sqrt(2^2 + 3^2 + 3^2)) = ceil(sqrt(22)) = 5
        // ceil(sqrt(2^2 + 2^2 + 1^2)) = ceil(sqrt(9)) = 3
        [
          [2, 3, 3],
          [2, 2, 1]
        ],

        // ceil(sqrt(6^2 + 2^2 + 3^2)) = ceil(sqrt(49)) = 7
        // ceil(sqrt(2^2 + 4^2 + 2^2)) = ceil(sqrt(24)) = 5
        [
          [6, 2, 3],
          [2, 4, 2]
        ]
      ]
    ]
  > : tensor<2x5x2x3xi8>

  // CHECK: MANP = 10 : ui{{[0-9]+}}
  %8 = "FHELinalg.matmul_int_eint"(%7, %0) : (tensor<2x5x2x3xi8>, tensor<3x2x!FHE.eint<7>>) -> tensor<2x5x2x2x!FHE.eint<7>>

  // ===============================

  return %2, %4, %6, %8 : tensor<2x!FHE.eint<7>>, tensor<2x2x!FHE.eint<7>>, tensor<5x2x2x!FHE.eint<7>>, tensor<2x5x2x2x!FHE.eint<7>>
}

// -----

func.func @matmul_int_eint_cst_different_operand_manp() -> tensor<2x!FHE.eint<7>> {
  %z = "FHE.zero_tensor"() : () -> tensor<3x2x!FHE.eint<7>>
  %a = arith.constant dense<[[4, 6], [2, 6], [5, 6]]> : tensor<3x2xi8>

  // CHECK: {MANP = 0 : ui{{[0-9]+}}}
  %0 = "FHELinalg.add_eint_int"(%z, %a) : (tensor<3x2x!FHE.eint<7>>, tensor<3x2xi8>) -> tensor<3x2x!FHE.eint<7>>

  // ===============================

  %1 = arith.constant dense<
    // ceil(sqrt(37 * (2^2 + 1^2 + 5^2) + 1)) = ceil(sqrt(31)) = 6
    [2, 1, 5]
  > : tensor<3xi8>

  // CHECK: MANP = 0 : ui{{[0-9]+}}
  %2 = "FHELinalg.matmul_int_eint"(%1, %0) : (tensor<3xi8>, tensor<3x2x!FHE.eint<7>>) -> tensor<2x!FHE.eint<7>>

  // ===============================

  return %2 : tensor<2x!FHE.eint<7>>
}

// -----



