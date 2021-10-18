// RUN: not zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'HLFHE.mul_eint_int' op  should have the width of plain input equals to width of encrypted input + 1
func @mul_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
  %0 = constant 1 : i4
  %1 = "HLFHE.mul_eint_int"(%arg0, %0): (!HLFHE.eint<2>, i4) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}
