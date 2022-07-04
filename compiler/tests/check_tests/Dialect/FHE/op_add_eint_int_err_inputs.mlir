// RUN: not concretecompiler --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.add_eint_int' op  should have the width of plain input equals to width of encrypted input + 1
func.func @add_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  %0 = arith.constant 1 : i4
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<2>, i4) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
