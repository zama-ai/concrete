// RUN: zamacompiler --passes MANP --action=dump-hlfhe --split-input-file %s 2>&1 | FileCheck %s

func @single_zero() -> !HLFHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "HLFHE.zero"() {MANP = 1 : ui{{[[0-9]+}}} : () -> !HLFHE.eint<2>
  %0 = "HLFHE.zero"() : () -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_cst_dot(%t: tensor<4x!HLFHE.eint<2>>) -> !HLFHE.eint<2>
{
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi3>

  // CHECK: %[[ret:.*]] = "HLFHE.dot_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 6 : ui{{[[0-9]+}}} : (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> !HLFHE.eint<2>
  %0 = "HLFHE.dot_eint_int"(%t, %cst) : (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_dyn_dot(%t: tensor<4x!HLFHE.eint<2>>, %dyn: tensor<4xi3>) -> !HLFHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "HLFHE.dot_eint_int"([[op0:.*]], %[[op1:.*]]) {MANP = 14 : ui{{[[0-9]+}}} : (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> !HLFHE.eint<2>
  %0 = "HLFHE.dot_eint_int"(%t, %dyn) : (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_cst_add_eint_int(%e: !HLFHE.eint<2>) -> !HLFHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // CHECK: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %0 = "HLFHE.add_eint_int"(%e, %cst) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_dyn_add_eint_int(%e: !HLFHE.eint<2>, %i: i3) -> !HLFHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %0 = "HLFHE.add_eint_int"(%e, %i) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_add_eint(%e0: !HLFHE.eint<2>, %e1: !HLFHE.eint<2>) -> !HLFHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "HLFHE.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  %0 = "HLFHE.add_eint"(%e0, %e1) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_cst_sub_int_eint(%e: !HLFHE.eint<2>) -> !HLFHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // CHECK: %[[ret:.*]] = "HLFHE.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (i3, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  %0 = "HLFHE.sub_int_eint"(%cst, %e) : (i3, !HLFHE.eint<2>) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_dyn_sub_int_eint(%e: !HLFHE.eint<2>, %i: i3) -> !HLFHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "HLFHE.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (i3, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  %0 = "HLFHE.sub_int_eint"(%i, %e) : (i3, !HLFHE.eint<2>) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_cst_mul_eint_int(%e: !HLFHE.eint<2>) -> !HLFHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // %0 = "HLFHE.mul_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 3 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %0 = "HLFHE.mul_eint_int"(%e, %cst) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_dyn_mul_eint_int(%e: !HLFHE.eint<2>, %i: i3) -> !HLFHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "HLFHE.mul_eint_int"([[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %0 = "HLFHE.mul_eint_int"(%e, %i) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>

  return %0 : !HLFHE.eint<2>
}

// -----

func @single_apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: tensor<4xi64>) -> !HLFHE.eint<2> {
  // CHECK: %[[ret:.*]] = "HLFHE.apply_lookup_table"(%[[op0:.*]], %[[op1:.*]]) {MANP = 1 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, tensor<4xi64>) -> !HLFHE.eint<2> 
  %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, tensor<4xi64>) -> !HLFHE.eint<2>
  return %1: !HLFHE.eint<2>
}

// -----

func @chain_add_eint_int(%e: !HLFHE.eint<2>) -> !HLFHE.eint<2>
{
  %cst0 = arith.constant 3 : i3
  %cst1 = arith.constant 7 : i3
  %cst2 = arith.constant 2 : i3
  %cst3 = arith.constant 1 : i3

  // CHECK: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %0 = "HLFHE.add_eint_int"(%e, %cst0) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %1 = "HLFHE.add_eint_int"(%0, %cst1) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %2 = "HLFHE.add_eint_int"(%1, %cst2) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %3 = "HLFHE.add_eint_int"(%2, %cst3) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>

  return %3 : !HLFHE.eint<2>
}

// -----

func @dag_add_eint_int(%e: !HLFHE.eint<2>) -> !HLFHE.eint<2>
{
  %Acst0 = arith.constant 3 : i3
  %Acst1 = arith.constant 7 : i3
  %Acst2 = arith.constant 2 : i3
  %Acst3 = arith.constant 1 : i3

  // CHECK: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %A0 = "HLFHE.add_eint_int"(%e, %Acst0) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>  
  %A1 = "HLFHE.add_eint_int"(%A0, %Acst1) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %A2 = "HLFHE.add_eint_int"(%A1, %Acst2) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %A3 = "HLFHE.add_eint_int"(%A2, %Acst3) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>

  %Bcst0 = arith.constant 1 : i3
  %Bcst1 = arith.constant 5 : i3
  %Bcst2 = arith.constant 2 : i3
  %Bcst3 = arith.constant 7 : i3
  %Bcst4 = arith.constant 4 : i3
  %Bcst5 = arith.constant 7 : i3

  // CHECK: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %B0 = "HLFHE.add_eint_int"(%e, %Bcst0) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 6 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %B1 = "HLFHE.add_eint_int"(%B0, %Bcst1) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 6 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %B2 = "HLFHE.add_eint_int"(%B1, %Bcst2) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %B3 = "HLFHE.add_eint_int"(%B2, %Bcst3) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 10 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %B4 = "HLFHE.add_eint_int"(%B3, %Bcst4) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 13 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  %B5 = "HLFHE.add_eint_int"(%B4, %Bcst5) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>

  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 15 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  %res = "HLFHE.add_eint"(%B5, %A3) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>

  return %A3 : !HLFHE.eint<2>
}

// -----

func @chain_add_eint(%e0: !HLFHE.eint<2>, %e1: !HLFHE.eint<2>, %e2: !HLFHE.eint<2>, %e3: !HLFHE.eint<2>, %e4: !HLFHE.eint<2>) -> !HLFHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "HLFHE.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  %0 = "HLFHE.add_eint"(%e0, %e1) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>

  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  %1 = "HLFHE.add_eint"(%0, %e2) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>

  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  %2 = "HLFHE.add_eint"(%1, %e3) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>

  // CHECK-NEXT: %[[ret:.*]] = "HLFHE.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 3 : ui{{[0-9]+}}} : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  %3 = "HLFHE.add_eint"(%2, %e4) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>

  return %3 : !HLFHE.eint<2>
}
