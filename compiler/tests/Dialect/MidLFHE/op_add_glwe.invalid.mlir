// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// GLWE p parameter result
func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{6}> {
  // expected-error @+1 {{'MidLFHE.add_glwe' op should have the same GLWE 'p' parameter}}
  %1 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,12,64}{6}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{6}>
}

// -----

// GLWE p parameter inputs
func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{1024,12,64}{6}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // expected-error @+1 {{'MidLFHE.add_glwe' op should have the same GLWE 'p' parameter}}
  %1 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{1024,12,64}{6}>) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{7}>
}

// -----

// GLWE dimension parameter result
func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{512,12,64}{7}> {
  // expected-error @+1 {{'MidLFHE.add_glwe' op should have the same GLWE 'dimension' parameter}}
  %1 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{512,12,64}{7}>)
  return %1: !MidLFHE.glwe<{512,12,64}{7}>
}

// -----

// GLWE dimension parameter inputs
func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{512,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // expected-error @+1 {{'MidLFHE.add_glwe' op should have the same GLWE 'dimension' parameter}}
  %1 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{512,12,64}{7}>) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{7}>
}

// -----

// GLWE polynomialSize parameter result
func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,11,64}{7}> {
  // expected-error @+1 {{'MidLFHE.add_glwe' op should have the same GLWE 'polynomialSize' parameter}}
  %1 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,11,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,11,64}{7}>
}

// -----

// GLWE polynomialSize parameter inputs
func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{1024,11,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // expected-error @+1 {{'MidLFHE.add_glwe' op should have the same GLWE 'polynomialSize' parameter}}
  %1 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{1024,11,64}{7}>) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{7}>
}

