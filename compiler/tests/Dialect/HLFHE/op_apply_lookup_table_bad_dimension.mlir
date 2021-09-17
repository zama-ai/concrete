// RUN: not zamacompiler --entry-dialect=hlfhe --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'HLFHE.apply_lookup_table' op  should have as `l_cst` argument a shape of one dimension equals to 2^p, where p is the width of the `ct` argument.
func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: tensor<8xi3>) -> !HLFHE.eint<2> {
  %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, tensor<8xi3>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}
