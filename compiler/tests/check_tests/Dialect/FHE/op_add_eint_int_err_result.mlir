// RUN: not concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.add_eint_int' op  should have the width of encrypted inputs and result equals
func @add_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<3> {
  %0 = arith.constant 1 : i2
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<2>, i2) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}
