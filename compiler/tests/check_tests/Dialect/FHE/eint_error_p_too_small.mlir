// RUN: not concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

// CHECK-LABEL: FHE.eint didn't support precision equals to 0
func.func @test(%arg0: !FHE.eint<0>) {
  return
}
