// RUN: concretecompiler --passes MANP --action=dump-fhe --split-input-file %s 2>&1 | FileCheck %s

func @single_zero() -> !FHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "FHE.zero"() {MANP = 1 : ui{{[[0-9]+}}} : () -> !FHE.eint<2>
  %0 = "FHE.zero"() : () -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_cst_add_eint_int(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // CHECK: %[[ret:.*]] = "FHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %0 = "FHE.add_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_dyn_add_eint_int(%e: !FHE.eint<2>, %i: i3) -> !FHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "FHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %0 = "FHE.add_eint_int"(%e, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_add_eint(%e0: !FHE.eint<2>, %e1: !FHE.eint<2>) -> !FHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "FHE.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  %0 = "FHE.add_eint"(%e0, %e1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_cst_sub_int_eint(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // CHECK: %[[ret:.*]] = "FHE.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (i3, !FHE.eint<2>) -> !FHE.eint<2>
  %0 = "FHE.sub_int_eint"(%cst, %e) : (i3, !FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_dyn_sub_int_eint(%e: !FHE.eint<2>, %i: i3) -> !FHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "FHE.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (i3, !FHE.eint<2>) -> !FHE.eint<2>
  %0 = "FHE.sub_int_eint"(%i, %e) : (i3, !FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_neg_eint(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "FHE.neg_eint"(%[[op0:.*]]) {MANP = 1 : ui{{[0-9]+}}} : (!FHE.eint<2>) -> !FHE.eint<2>
  %0 = "FHE.neg_eint"(%e) : (!FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_cst_mul_eint_int(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // %0 = "FHE.mul_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 3 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %0 = "FHE.mul_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_dyn_mul_eint_int(%e: !FHE.eint<2>, %i: i3) -> !FHE.eint<2>
{
  // CHECK: %[[ret:.*]] = "FHE.mul_eint_int"([[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %0 = "FHE.mul_eint_int"(%e, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<2> {
  // CHECK: %[[ret:.*]] = "FHE.apply_lookup_table"(%[[op0:.*]], %[[op1:.*]]) {MANP = 1 : ui{{[0-9]+}}} : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2> 
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  return %1: !FHE.eint<2>
}

// -----

func @chain_add_eint_int(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst0 = arith.constant 3 : i3
  %cst1 = arith.constant 7 : i3
  %cst2 = arith.constant 2 : i3
  %cst3 = arith.constant 1 : i3

  // CHECK: %[[V0:.*]] = "FHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %0 = "FHE.add_eint_int"(%e, %cst0) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.add_eint_int"(%[[V0]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %1 = "FHE.add_eint_int"(%0, %cst1) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V2:.*]] = "FHE.add_eint_int"(%[[V1]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %2 = "FHE.add_eint_int"(%1, %cst2) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint_int"(%[[V2]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %3 = "FHE.add_eint_int"(%2, %cst3) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %3 : !FHE.eint<2>
}

// -----

func @dag_add_eint_int(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %Acst0 = arith.constant 3 : i3
  %Acst1 = arith.constant 7 : i3
  %Acst2 = arith.constant 2 : i3
  %Acst3 = arith.constant 1 : i3

  // CHECK: %[[V0:.*]] = "FHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %A0 = "FHE.add_eint_int"(%e, %Acst0) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.add_eint_int"(%[[V0]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>  
  %A1 = "FHE.add_eint_int"(%A0, %Acst1) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V2:.*]] = "FHE.add_eint_int"(%[[V1]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %A2 = "FHE.add_eint_int"(%A1, %Acst2) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint_int"(%[[V2]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %A3 = "FHE.add_eint_int"(%A2, %Acst3) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  %Bcst0 = arith.constant 1 : i3
  %Bcst1 = arith.constant 5 : i3
  %Bcst2 = arith.constant 2 : i3
  %Bcst3 = arith.constant 7 : i3
  %Bcst4 = arith.constant 4 : i3
  %Bcst5 = arith.constant 7 : i3

  // CHECK: %[[V0:.*]] = "FHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %B0 = "FHE.add_eint_int"(%e, %Bcst0) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.add_eint_int"(%[[V0]], %[[op1:.*]]) {MANP = 6 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %B1 = "FHE.add_eint_int"(%B0, %Bcst1) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V2:.*]] = "FHE.add_eint_int"(%[[V1]], %[[op1:.*]]) {MANP = 6 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %B2 = "FHE.add_eint_int"(%B1, %Bcst2) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint_int"(%[[V2]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %B3 = "FHE.add_eint_int"(%B2, %Bcst3) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V4:.*]] = "FHE.add_eint_int"(%[[V3]], %[[op1:.*]]) {MANP = 10 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %B4 = "FHE.add_eint_int"(%B3, %Bcst4) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V5:.*]] = "FHE.add_eint_int"(%[[V4]], %[[op1:.*]]) {MANP = 13 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %B5 = "FHE.add_eint_int"(%B4, %Bcst5) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  // CHECK-NEXT: %[[V6:.*]] = "FHE.add_eint"(%[[V5]], %[[op1:.*]]) {MANP = 15 : ui{{[0-9]+}}} : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  %res = "FHE.add_eint"(%B5, %A3) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  return %A3 : !FHE.eint<2>
}

// -----

func @chain_add_eint(%e0: !FHE.eint<2>, %e1: !FHE.eint<2>, %e2: !FHE.eint<2>, %e3: !FHE.eint<2>, %e4: !FHE.eint<2>) -> !FHE.eint<2>
{
  // CHECK: %[[V0:.*]] = "FHE.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  %0 = "FHE.add_eint"(%e0, %e1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK-NEXT: %[[V1:.*]] = "FHE.add_eint"(%[[V0]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  %1 = "FHE.add_eint"(%0, %e2) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK-NEXT: %[[V2:.*]] = "FHE.add_eint"(%[[V1]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  %2 = "FHE.add_eint"(%1, %e3) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint"(%[[V2]], %[[op1:.*]]) {MANP = 3 : ui{{[0-9]+}}} : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  %3 = "FHE.add_eint"(%2, %e4) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  return %3 : !FHE.eint<2>
}


// -----

func @chain_add_eint_neg_eint(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst0 = arith.constant 3 : i3

  // CHECK: %[[V0:.*]] = "FHE.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  %0 = "FHE.add_eint_int"(%e, %cst0) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[ret:.*]] = "FHE.neg_eint"(%[[V0]]) {MANP = 4 : ui{{[0-9]+}}} : (!FHE.eint<2>) -> !FHE.eint<2>
  %1 = "FHE.neg_eint"(%0) : (!FHE.eint<2>) -> !FHE.eint<2>

  return %1 : !FHE.eint<2>
}