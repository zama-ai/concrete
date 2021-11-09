// RUN: zamacompiler --passes MANP --action=dump-hlfhe --split-input-file %s 2>&1 | FileCheck %s

func @single_cst_add_eint_int(%t: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint_int"(%t, %cst) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_dyn_add_eint_int(%e: tensor<8x!HLFHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint_int"(%e, %i) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_add_eint(%e0: tensor<8x!HLFHE.eint<2>>, %e1: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint"(%e0, %e1) : (tensor<8x!HLFHE.eint<2>>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_cst_sub_int_eint(%e: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: %[[ret:.*]] = "HLFHELinalg.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8xi3>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.sub_int_eint"(%cst, %e) : (tensor<8xi3>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_neg_eint(%e: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.neg_eint"(%[[op0:.*]]) {MANP = 1 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.neg_eint"(%e) : (tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_dyn_sub_int_eint(%e: tensor<8x!HLFHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (tensor<8xi3>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.sub_int_eint"(%i, %e) : (tensor<8xi3>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_cst_mul_eint_int(%e: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // %0 = "HLFHELinalg.mul_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 3 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.mul_eint_int"(%e, %cst) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_dyn_mul_eint_int(%e: tensor<8x!HLFHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.mul_eint_int"([[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.mul_eint_int"(%e, %i) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @chain_add_eint_int(%e: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst0 = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>
  %cst1 = arith.constant dense<[0, 7, 2, 5, 6, 2, 1, 7]> : tensor<8xi3>
  %cst2 = arith.constant dense<[0, 1, 2, 0, 1, 2, 0, 1]> : tensor<8xi3>
  %cst3 = arith.constant dense<[0, 1, 1, 0, 0, 1, 0, 1]> : tensor<8xi3>
  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint_int"(%e, %cst0) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %1 = "HLFHELinalg.add_eint_int"(%0, %cst1) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %2 = "HLFHELinalg.add_eint_int"(%1, %cst2) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %3 = "HLFHELinalg.add_eint_int"(%2, %cst3) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  return %3 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @chain_add_eint_int_neg_eint(%e: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst0 = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>
  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint_int"(%e, %cst0) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHELinalg.neg_eint"(%[[op0:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  %1 = "HLFHELinalg.neg_eint"(%0) : (tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  return %1 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @apply_lookup_table(%t: tensor<3x3x!HLFHE.eint<2>>) -> tensor<3x3x!HLFHE.eint<3>> {
  %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
  // CHECK: %[[RES:.*]] = "HLFHELinalg.apply_lookup_table"(%[[T:.*]], %[[LUT:.*]]) {MANP = 1 : ui1} : (tensor<3x3x!HLFHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!HLFHE.eint<3>>
  %res = "HLFHELinalg.apply_lookup_table"(%t, %lut) : (tensor<3x3x!HLFHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!HLFHE.eint<3>>
  return %res : tensor<3x3x!HLFHE.eint<3>>
}

// -----

func @apply_lookup_table_after_op(%t: tensor<8x!HLFHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!HLFHE.eint<3>> {
  %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
  // CHECK: %[[V0:.*]] = "HLFHELinalg.mul_eint_int"([[T:.*]], %[[I:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.mul_eint_int"(%t, %i) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[RES:.*]] = "HLFHELinalg.apply_lookup_table"(%[[V0:.*]], %[[LUT:.*]]) {MANP = 1 : ui1} : (tensor<8x!HLFHE.eint<2>>, tensor<4xi64>) -> tensor<8x!HLFHE.eint<3>>
  %res = "HLFHELinalg.apply_lookup_table"(%0, %lut) : (tensor<8x!HLFHE.eint<2>>, tensor<4xi64>) -> tensor<8x!HLFHE.eint<3>>
  return %res : tensor<8x!HLFHE.eint<3>>
}

// -----

func @matmul_eint_int_dyn_p_1(%arg0: tensor<3x1x!HLFHE.eint<2>>, %arg1: tensor<1x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 1 = 65
  // ceil(sqrt(65)) = 9
  // CHECK: %[[V1:.*]] = "HLFHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 9 : ui{{[0-9]+}}} 
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x1x!HLFHE.eint<2>>, tensor<1x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

// -----

func @matmul_eint_int_dyn_p_2(%arg0: tensor<3x2x!HLFHE.eint<2>>, %arg1: tensor<2x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 1 = 65
  // p = 1
  // manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 65 = 129
  // ceil(sqrt(129)) = 12
  // CHECK: %[[V1:.*]] = "HLFHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 12 : ui{{[0-9]+}}}
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x2x!HLFHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

// -----

func @matmul_eint_int_cst_p_1(%arg0: tensor<3x1x!HLFHE.eint<2>>) -> tensor<3x2x!HLFHE.eint<2>> {
  %0 = arith.constant dense<[[3, 1]]> : tensor<1x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max cst is used for n = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 1 = 10
  // ceil(sqrt(10)) = 4
  // CHECK: %[[V1:.*]] = "HLFHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 4 : ui{{[0-9]+}}}
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x1x!HLFHE.eint<2>>, tensor<1x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

// -----

func @matmul_eint_int_cst_p_2_n_0(%arg0: tensor<3x2x!HLFHE.eint<2>>) -> tensor<3x2x!HLFHE.eint<2>> {
  %0 = arith.constant dense<[[4, 1],[3, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,3] are used for n = 0
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 1 = 10
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 17
  // manp(add_eint(mul, acc)) = 17 + 9 = 26
  // ceil(sqrt(26)) = 6
  // CHECK: %[[V1:.*]] = "HLFHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 6 : ui{{[0-9]+}}}
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x2x!HLFHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

// -----

func @matmul_eint_int_cst_p_2_n_1(%arg0: tensor<3x2x!HLFHE.eint<2>>) -> tensor<3x2x!HLFHE.eint<2>> {
  %0 = arith.constant dense<[[1, 4],[3, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,1] are used for n = 1
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 16
  // manp(add_eint(mul, acc)) = 16 + 1 = 17
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 1) = 1 * 1^2 = 1
  // manp(add_eint(mul, acc)) = 1 + 17 = 18
  // ceil(sqrt(18)) = 5
  // CHECK: %[[V1:.*]] = "HLFHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 5 : ui{{[0-9]+}}}
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x2x!HLFHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

// -----

func @apply_multi_lookup_table(%t: tensor<3x3x!HLFHE.eint<2>>, %luts: tensor<3x3x4xi64>) -> tensor<3x3x!HLFHE.eint<3>> {
  // CHECK: %[[RES:.*]] = "HLFHELinalg.apply_multi_lookup_table"(%[[T:.*]], %[[LUT:.*]]) {MANP = 1 : ui1} : (tensor<3x3x!HLFHE.eint<2>>, tensor<3x3x4xi64>) -> tensor<3x3x!HLFHE.eint<3>>
  %res = "HLFHELinalg.apply_multi_lookup_table"(%t, %luts) : (tensor<3x3x!HLFHE.eint<2>>, tensor<3x3x4xi64>) -> tensor<3x3x!HLFHE.eint<3>>
  return %res : tensor<3x3x!HLFHE.eint<3>>
}

// -----

func @apply_multi_lookup_table_after_op(%t: tensor<8x!HLFHE.eint<2>>, %i: tensor<8xi3>, %luts: tensor<8x4xi64>) -> tensor<8x!HLFHE.eint<3>> {
  // CHECK: %[[V0:.*]] = "HLFHELinalg.mul_eint_int"([[T:.*]], %[[I:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.mul_eint_int"(%t, %i) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[RES:.*]] = "HLFHELinalg.apply_multi_lookup_table"(%[[V0:.*]], %[[LUT:.*]]) {MANP = 1 : ui1} : (tensor<8x!HLFHE.eint<2>>, tensor<8x4xi64>) -> tensor<8x!HLFHE.eint<3>>
  %res = "HLFHELinalg.apply_multi_lookup_table"(%0, %luts) : (tensor<8x!HLFHE.eint<2>>, tensor<8x4xi64>) -> tensor<8x!HLFHE.eint<3>>
  return %res : tensor<8x!HLFHE.eint<3>>
}