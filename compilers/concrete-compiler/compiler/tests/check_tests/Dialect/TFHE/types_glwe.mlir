// RUN: concretecompiler %s --action=roundtrip 2>&1| FileCheck %s

// CHECK-LABEL: func.func @glwe_0(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
func.func @glwe_0(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
  // CHECK-LABEL: return %arg0 : !TFHE.glwe<sk[1]<12,1024>>
  return %arg0: !TFHE.glwe<sk[1]<12,1024>>
}

// -----

// CHECK-LABEL: func.func @glwe_1(%arg0: !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
func.func @glwe_1(%arg0: !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?> {
  // CHECK-LABEL: return %arg0 : !TFHE.glwe<sk?>
  return %arg0: !TFHE.glwe<sk?>
}
