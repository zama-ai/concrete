// RUN: not zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'HLFHE.sub_int_eint' op  should have the width of encrypted inputs and result equals
func @sub_int_eint(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<3> {
  %0 = arith.constant 1 : i2
  %1 = "HLFHE.sub_int_eint"(%0, %arg0): (i2, !HLFHE.eint<2>) -> (!HLFHE.eint<3>)
  return %1: !HLFHE.eint<3>
}
