// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.to_unsigned' op should have the width of encrypted input and result equal
func.func @bad_result_width(%arg0: !FHE.esint<2>) -> !FHE.eint<3> {
  %1 = "FHE.to_unsigned"(%arg0): (!FHE.esint<2>) -> !FHE.eint<3>
  return %1: !FHE.eint<3>
}
