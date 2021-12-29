// RUN: not concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.sub_int_eint' op  should have the width of plain input equals to width of encrypted input + 1
func @sub_int_eint(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  %0 = arith.constant 1 : i4
  %1 = "FHE.sub_int_eint"(%0, %arg0): (i4, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
