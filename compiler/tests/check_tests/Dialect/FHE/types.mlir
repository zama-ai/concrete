// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

// CHECK-LABEL: func.func @memref_arg(%arg0: memref<2x!FHE.eint<7>>
func.func @memref_arg(%arg0: memref<2x!FHE.eint<7>>) {
  return
}
