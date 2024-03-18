// RUN: concretecompiler --passes MANP --passes ConcreteOptimizer --optimizer-strategy=dag-mono --action=dump-fhe-no-linalg --split-input-file %s 2>&1 | FileCheck %s

func.func @single_zero() -> !FHE.eint<2>
{
  // CHECK: MANP = 0 : ui{{[0-9]+}}
  %0 = "FHE.zero"() : () -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @zero() -> tensor<8x!FHE.eint<2>>
{
  // CHECK: MANP = 0 : ui{{[0-9]+}}
  %0 = "FHE.zero_tensor"() : () -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func.func @single_cst_add_eint_int(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.add_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_add_eint_int_neg(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant -3 : i3

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.add_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_dyn_add_eint_int(%e: !FHE.eint<2>, %i: i3) -> !FHE.eint<2>
{
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.add_eint_int"(%e, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_add_eint(%e0: !FHE.eint<2>, %e1: !FHE.eint<2>) -> !FHE.eint<2>
{
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %0 = "FHE.add_eint"(%e0, %e1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_sub_int_eint(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.sub_int_eint"(%cst, %e) : (i3, !FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_sub_int_eint_neg(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant -3 : i3

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.sub_int_eint"(%cst, %e) : (i3, !FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_dyn_sub_int_eint(%e: !FHE.eint<2>, %i: i3) -> !FHE.eint<2>
{
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.sub_int_eint"(%i, %e) : (i3, !FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_sub_eint_int(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.sub_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_sub_eint_int_neg(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant -3 : i3

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.sub_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_dyn_sub_eint_int(%e: !FHE.eint<2>, %i: i3) -> !FHE.eint<2>
{
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.sub_eint_int"(%e, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @chain_sub_eint(%e0: !FHE.eint<2>, %e1: !FHE.eint<2>, %e2: !FHE.eint<2>, %e3: !FHE.eint<2>, %e4: !FHE.eint<2>) -> !FHE.eint<2>
{
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %0 = "FHE.sub_eint"(%e0, %e1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %1 = "FHE.sub_eint"(%0, %e2) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %2 = "FHE.sub_eint"(%1, %e3) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %3 = "FHE.sub_eint"(%2, %e4) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  return %3 : !FHE.eint<2>
}

// -----

func.func @single_neg_eint(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.neg_eint"(%e) : (!FHE.eint<2>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_mul_eint_int(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant 3 : i3

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHE.mul_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_mul_eint_int_neg(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant -3 : i3

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHE.mul_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_cst_mul_eint_int_neg(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst = arith.constant -1 : i3

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.mul_eint_int"(%e, %cst) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_dyn_mul_eint_int(%e: !FHE.eint<2>, %i: i3) -> !FHE.eint<2>
{
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHE.mul_eint_int"(%e, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func.func @single_dyn_mul_eint_int_6b_3b(%e: !FHE.eint<6>, %i: i3) -> !FHE.eint<6>
{
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %0 = "FHE.mul_eint_int"(%e, %i) : (!FHE.eint<6>, i3) -> !FHE.eint<6>

  return %0 : !FHE.eint<6>
}

// -----

func.func @single_dyn_mul_eint_int_6b_4b(%e: !FHE.eint<6>, %i: i4) -> !FHE.eint<6>
{
  // CHECK: MANP = 7 : ui{{[0-9]+}}
  %0 = "FHE.mul_eint_int"(%e, %i) : (!FHE.eint<6>, i4) -> !FHE.eint<6>

  return %0 : !FHE.eint<6>
}

// -----

func.func @single_dyn_mul_eint_int_6b_5b(%e: !FHE.eint<6>, %i: i5) -> !FHE.eint<6>
{
  // CHECK: MANP = 15 : ui{{[0-9]+}}
  %0 = "FHE.mul_eint_int"(%e, %i) : (!FHE.eint<6>, i5) -> !FHE.eint<6>

  return %0 : !FHE.eint<6>
}

// -----

func.func @single_apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<2> {
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  return %1: !FHE.eint<2>
}

// -----

func.func @chain_add_eint_int(%e: !FHE.eint<3>) -> !FHE.eint<3>
{
  %cst0 = arith.constant 3 : i4
  %cst1 = arith.constant 7 : i4
  %cst2 = arith.constant 2 : i4
  %cst3 = arith.constant 1 : i4

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.add_eint_int"(%e, %cst0) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %1 = "FHE.add_eint_int"(%0, %cst1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %2 = "FHE.add_eint_int"(%1, %cst2) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %3 = "FHE.add_eint_int"(%2, %cst3) : (!FHE.eint<3>, i4) -> !FHE.eint<3>

  return %3 : !FHE.eint<3>
}

// -----

func.func @dag_add_eint_int(%e: !FHE.eint<3>) -> !FHE.eint<3>
{
  %Acst0 = arith.constant 3 : i4
  %Acst1 = arith.constant 7 : i4
  %Acst2 = arith.constant 2 : i4
  %Acst3 = arith.constant 1 : i4

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %A0 = "FHE.add_eint_int"(%e, %Acst0) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %A1 = "FHE.add_eint_int"(%A0, %Acst1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %A2 = "FHE.add_eint_int"(%A1, %Acst2) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %A3 = "FHE.add_eint_int"(%A2, %Acst3) : (!FHE.eint<3>, i4) -> !FHE.eint<3>

  %Bcst0 = arith.constant 1 : i4
  %Bcst1 = arith.constant 5 : i4
  %Bcst2 = arith.constant 2 : i4
  %Bcst3 = arith.constant 7 : i4
  %Bcst4 = arith.constant 4 : i4
  %Bcst5 = arith.constant 7 : i4

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %B0 = "FHE.add_eint_int"(%e, %Bcst0) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %B1 = "FHE.add_eint_int"(%B0, %Bcst1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %B2 = "FHE.add_eint_int"(%B1, %Bcst2) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %B3 = "FHE.add_eint_int"(%B2, %Bcst3) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %B4 = "FHE.add_eint_int"(%B3, %Bcst4) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %B5 = "FHE.add_eint_int"(%B4, %Bcst5) : (!FHE.eint<3>, i4) -> !FHE.eint<3>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %res = "FHE.add_eint"(%B5, %A3) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>

  return %A3 : !FHE.eint<3>
}

// -----

func.func @chain_add_eint(%e0: !FHE.eint<2>, %e1: !FHE.eint<2>, %e2: !FHE.eint<2>, %e3: !FHE.eint<2>, %e4: !FHE.eint<2>) -> !FHE.eint<2>
{
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %0 = "FHE.add_eint"(%e0, %e1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %1 = "FHE.add_eint"(%0, %e2) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %2 = "FHE.add_eint"(%1, %e3) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %3 = "FHE.add_eint"(%2, %e4) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>

  return %3 : !FHE.eint<2>
}


// -----

func.func @chain_add_eint_neg_eint(%e: !FHE.eint<2>) -> !FHE.eint<2>
{
  %cst0 = arith.constant 3 : i3

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %0 = "FHE.add_eint_int"(%e, %cst0) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %1 = "FHE.neg_eint"(%0) : (!FHE.eint<2>) -> !FHE.eint<2>

  return %1 : !FHE.eint<2>
}

// -----

// CHECK-LABEL:  @transpose_eint_3D(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>>
func.func @transpose_eint_3D(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>> {
  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %c = "FHELinalg.transpose"(%arg0) : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>>
  return %c : tensor<5x4x3x!FHE.eint<6>>
}

// -----

// CHECK-LABEL:  @transpose_eint_3D_after_op(%arg0: tensor<3x4x5x!FHE.eint<6>>, %arg1: tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>>
func.func @transpose_eint_3D_after_op(%arg0: tensor<3x4x5x!FHE.eint<6>>, %arg1: tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>> {
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %sum = "FHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x4x5x!FHE.eint<6>>, tensor<3x4x5x!FHE.eint<6>>) -> tensor<3x4x5x!FHE.eint<6>>
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %c = "FHELinalg.transpose"(%sum) : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>>
  return %c : tensor<5x4x3x!FHE.eint<6>>
}
