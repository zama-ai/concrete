// RUN: not zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'HLFHE.neg_eint' op  should have the width of encrypted inputs and result equals
func @sub_int_eint(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<3> {
  %1 = "HLFHE.neg_eint"(%arg0): (!HLFHE.eint<2>) -> (!HLFHE.eint<3>)
  return %1: !HLFHE.eint<3>
}
