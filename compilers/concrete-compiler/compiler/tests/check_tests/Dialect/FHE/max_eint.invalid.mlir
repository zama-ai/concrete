// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// -----

// CHECK-LABEL: error: 'FHE.max_eint' op should have the width of encrypted inputs equal
func.func @bad_inputs_width(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<3>) -> !FHE.eint<5> {
  %1 = "FHE.max_eint"(%arg0, %arg1): (!FHE.eint<5>, !FHE.eint<3>) -> (!FHE.eint<5>)
  return %1: !FHE.eint<5>
}

// -----

// CHECK-LABEL: error: 'FHE.max_eint' op should have the signedness of encrypted inputs equal
func.func @bad_inputs_signedness(%arg0: !FHE.eint<5>, %arg1: !FHE.esint<5>) -> !FHE.eint<5> {
  %1 = "FHE.max_eint"(%arg0, %arg1): (!FHE.eint<5>, !FHE.esint<5>) -> (!FHE.eint<5>)
  return %1: !FHE.eint<5>
}

// -----

// CHECK-LABEL: error: 'FHE.max_eint' op should have the width of encrypted inputs and result equal
func.func @bad_result_width(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<3> {
  %1 = "FHE.max_eint"(%arg0, %arg1): (!FHE.eint<5>, !FHE.eint<5>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

// -----

// CHECK-LABEL: error: 'FHE.max_eint' op should have the signedness of encrypted inputs and result equal
func.func @bad_result_signedness(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.esint<5> {
  %1 = "FHE.max_eint"(%arg0, %arg1): (!FHE.eint<5>, !FHE.eint<5>) -> (!FHE.esint<5>)
  return %1: !FHE.esint<5>
}
