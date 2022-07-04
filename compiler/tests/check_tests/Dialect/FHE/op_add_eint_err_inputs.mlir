// RUN: not concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.add_eint' op  should have the width of encrypted inputs equals
func.func @add_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<3>) -> !FHE.eint<2> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<3>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
