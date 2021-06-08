// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @glwe_unknwon_parameter(%arg0: !MidLFHE.glwe<{_,_,_}{_,7,_,_,_}>) -> !MidLFHE.glwe<{_,_,_}{_,7,_,_,_}>
func @glwe_unknwon_parameter(%arg0: !MidLFHE.glwe<{_,_,_}{_,7,_,_,_}>) -> !MidLFHE.glwe<{_,_,_}{_,7,_,_,_}> {
  // CHECK-LABEL: return %arg0 : !MidLFHE.glwe<{_,_,_}{_,7,_,_,_}>
  return %arg0: !MidLFHE.glwe<{_,_,_}{_,7,_,_,_}>
}

// CHECK-LABEL: func @glwe(%arg0: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>) -> !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
func @glwe(%arg0: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>) -> !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}> {
  // CHECK-LABEL: return %arg0 : !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
  return %arg0: !MidLFHE.glwe<{1024,12,-64}{0,7,0,32,-25}>
}