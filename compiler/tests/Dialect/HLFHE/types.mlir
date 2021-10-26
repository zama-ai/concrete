// RUN: zamacompiler --action=roundtrip %s  2>&1| FileCheck %s

// CHECK-LABEL: func @memref_arg(%arg0: memref<2x!HLFHE.eint<7>>
func @memref_arg(%arg0: memref<2x!HLFHE.eint<7>>) {
  return
}
