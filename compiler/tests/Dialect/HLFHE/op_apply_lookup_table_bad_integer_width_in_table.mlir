// RUN: not zamacompiler %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'HLFHE.apply_lookup_table' op  should have equals width beetwen the encrypted integer result and integers of the `tabulated_lambda` argument
func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: memref<4xi3>) -> !HLFHE.eint<2> {
  %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, memref<4xi3>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}