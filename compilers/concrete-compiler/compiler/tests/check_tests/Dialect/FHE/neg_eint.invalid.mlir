// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.neg_eint' op should have the width of encrypted inputs and result equal
func.func @bad_result_width(%arg0: !FHE.eint<2>) -> !FHE.eint<3> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<2>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

// -----

// CHECK-LABEL: error: 'FHE.neg_eint' op should have the signedness of encrypted inputs and result equal
func.func @bad_result_signedness(%arg0: !FHE.eint<2>) -> !FHE.esint<2> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<2>) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
}
