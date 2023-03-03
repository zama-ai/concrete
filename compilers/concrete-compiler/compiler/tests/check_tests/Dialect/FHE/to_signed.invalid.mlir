// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.to_signed' op should have the width of encrypted input and result equal
func.func @bad_result_width(%arg0: !FHE.eint<2>) -> !FHE.esint<3> {
  %1 = "FHE.to_signed"(%arg0): (!FHE.eint<2>) -> !FHE.esint<3>
  return %1: !FHE.esint<3>
}
