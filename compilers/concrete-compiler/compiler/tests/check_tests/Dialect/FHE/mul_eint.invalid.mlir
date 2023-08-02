// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.mul_eint' op should have the width of encrypted inputs equal
func.func @bad_inputs_width(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<3>) -> !FHE.eint<2> {
  %1 = "FHE.mul_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<3>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
