// RUN: zamacompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// GLWE p parameter
func @neg_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{6}> {
  // expected-error @+1 {{'MidLFHE.neg_glwe' op should have the same GLWE 'p' parameter}}
  %1 = "MidLFHE.neg_glwe"(%arg0): (!MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,12,64}{6}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{6}>
}

// -----

// GLWE dimension parameter
func @neg_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{512,12,64}{7}> {
  // expected-error @+1 {{'MidLFHE.neg_glwe' op should have the same GLWE 'dimension' parameter}}
  %1 = "MidLFHE.neg_glwe"(%arg0): (!MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{512,12,64}{7}>)
  return %1: !MidLFHE.glwe<{512,12,64}{7}>
}

// -----

// GLWE polynomialSize parameter
func @neg_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,11,64}{7}> {
  // expected-error @+1 {{'MidLFHE.neg_glwe' op should have the same GLWE 'polynomialSize' parameter}}
  %1 = "MidLFHE.neg_glwe"(%arg0): (!MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,11,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,11,64}{7}>
}

// -----

// integer width doesn't match GLWE parameter
func @neg_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,11,64}{7}> {
  // expected-error @+1 {{'MidLFHE.neg_glwe' op should have the same GLWE 'polynomialSize' parameter}}
  %1 = "MidLFHE.neg_glwe"(%arg0): (!MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,11,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,11,64}{7}>
}

