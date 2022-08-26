// RUN: not concretecompiler --split-input-file --action=roundtrip %s  2>&1| FileCheck %s

// CHECK-LABEL: FHE.eint doesn't support precision of 0
func.func @test(%arg0: !FHE.eint<0>) {
  return
}

// -----

// CHECK-LABEL: FHE.esint doesn't support precision of 0
func.func @test_signed(%arg0: !FHE.esint<0>) {
  return
}
