// RUN: not zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'HLFHE.apply_lookup_table' op : `l_cst` (operand #2) inner dimension should have size 4(=2^2) to match `ct` (operand #1) elements bitwidth (2)
func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: tensor<8xi3>) -> !HLFHE.eint<2> {
  %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, tensor<8xi3>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}
