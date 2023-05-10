// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.round' op should have the input width larger than the output width.
func.func @larger_output_width(%arg0: !FHE.eint<3>) -> !FHE.eint<4> {
  %1 = "FHE.round"(%arg0): (!FHE.eint<3>) -> (!FHE.eint<4>)
  return %1: !FHE.eint<4>
}

// -----

// CHECK-LABEL: error: 'FHE.round' op should have the signedness of encrypted inputs and result equal
func.func @signed_input(%arg0: !FHE.esint<3>) -> !FHE.eint<2> {
  %1 = "FHE.round"(%arg0): (!FHE.esint<3>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
