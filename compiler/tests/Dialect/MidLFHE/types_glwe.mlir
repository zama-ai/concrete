// RUN: zamacompiler %s --entry-dialect=midlfhe --action=roundtrip 2>&1| FileCheck %s

// CHECK-LABEL: func @glwe_0(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
func @glwe_0(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // CHECK-LABEL: return %arg0 : !MidLFHE.glwe<{1024,12,64}{7}>
  return %arg0: !MidLFHE.glwe<{1024,12,64}{7}>
}

// CHECK-LABEL: func @glwe_1(%arg0: !MidLFHE.glwe<{_,_,_}{7}>) -> !MidLFHE.glwe<{_,_,_}{7}>
func @glwe_1(%arg0: !MidLFHE.glwe<{_,_,_}{7}>) -> !MidLFHE.glwe<{_,_,_}{7}> {
  // CHECK-LABEL: return %arg0 : !MidLFHE.glwe<{_,_,_}{7}>
  return %arg0: !MidLFHE.glwe<{_,_,_}{7}>
}
