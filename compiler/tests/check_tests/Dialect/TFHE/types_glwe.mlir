// RUN: concretecompiler %s --action=roundtrip 2>&1| FileCheck %s

// CHECK-LABEL: func.func @glwe_0(%arg0: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}>
func.func @glwe_0(%arg0: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}> {
  // CHECK-LABEL: return %arg0 : !TFHE.glwe<{1024,12,64}{7}>
  return %arg0: !TFHE.glwe<{1024,12,64}{7}>
}

// CHECK-LABEL: func.func @glwe_1(%arg0: !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
func.func @glwe_1(%arg0: !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}> {
  // CHECK-LABEL: return %arg0 : !TFHE.glwe<{_,_,_}{7}>
  return %arg0: !TFHE.glwe<{_,_,_}{7}>
}

// CHECK-LABEL: func.func @glwe_crt(%arg0: !TFHE.glwe<crt=[2,3,5,7,11]{_,_,_}{7}>) -> !TFHE.glwe<crt=[2,3,5,7,11]{_,_,_}{7}>
func.func @glwe_crt(%arg0: !TFHE.glwe<crt=[2,3,5,7,11]{_,_,_}{7}>) -> !TFHE.glwe<crt=[2,3,5,7,11]{_,_,_}{7}> {
  // CHECK-LABEL: return %arg0 : !TFHE.glwe<crt=[2,3,5,7,11]{_,_,_}{7}>
  return %arg0: !TFHE.glwe<crt=[2,3,5,7,11]{_,_,_}{7}>
}

// CHECK-LABEL: func.func @glwe_crt_undef(%arg0: !TFHE.glwe<crt=[_,_,_,_,_]{_,_,_}{7}>) -> !TFHE.glwe<crt=[_,_,_,_,_]{_,_,_}{7}>
func.func @glwe_crt_undef(%arg0: !TFHE.glwe<crt=[_,_,_,_,_]{_,_,_}{7}>) -> !TFHE.glwe<crt=[_,_,_,_,_]{_,_,_}{7}> {
  // CHECK-LABEL: return %arg0 : !TFHE.glwe<crt=[_,_,_,_,_]{_,_,_}{7}>
  return %arg0: !TFHE.glwe<crt=[_,_,_,_,_]{_,_,_}{7}>
}
