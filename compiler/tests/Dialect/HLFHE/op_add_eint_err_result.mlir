// RUN: not zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'HLFHE.add_eint' op  should have the width of encrypted inputs and result equals
func @add_eint(%arg0: !HLFHE.eint<2>, %arg1: !HLFHE.eint<2>) -> !HLFHE.eint<3> {
  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<2>, !HLFHE.eint<2>) -> (!HLFHE.eint<3>)
  return %1: !HLFHE.eint<3>
}
