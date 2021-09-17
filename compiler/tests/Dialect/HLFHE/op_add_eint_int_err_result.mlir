// RUN: not zamacompiler --entry-dialect=hlfhe --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'HLFHE.add_eint_int' op  should have the width of encrypted inputs and result equals
func @add_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<3> {
  %0 = constant 1 : i2
  %1 = "HLFHE.add_eint_int"(%arg0, %0): (!HLFHE.eint<2>, i2) -> (!HLFHE.eint<3>)
  return %1: !HLFHE.eint<3>
}
