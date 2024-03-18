// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.mul_eint_int' op should have the width of encrypted inputs and result equal
func.func @bad_result_width(%arg0: !FHE.eint<2>) -> !FHE.eint<3> {
  %0 = arith.constant 1 : i3
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

// -----

// CHECK-LABEL: error: 'FHE.mul_eint_int' op should have the signedness of encrypted inputs and result equal
func.func @bad_result_signedness(%arg0: !FHE.eint<2>) -> !FHE.esint<2> {
  %0 = arith.constant 1 : i3
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
}
