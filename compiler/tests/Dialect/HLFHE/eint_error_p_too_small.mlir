// RUN: not zamacompiler --entry-dialect=hlfhe --action=roundtrip %s  2>&1| FileCheck %s

// CHECK-LABEL: eint support only precision in ]0;7]
func @test(%arg0: !HLFHE.eint<0>) {
  return
}
