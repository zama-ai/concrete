// RUN: not concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.apply_lookup_table' op : `lut` (operand #2) inner dimension should have size 4(=2^2) to match `ct` (operand #1) elements bitwidth (2)
func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<8xi3>) -> !FHE.eint<2> {
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<8xi3>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}
