// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// GLWE dimension parameter result
func.func @add_glwe(%arg0: !TFHE.glwe<{1024,12,64}{7}>, %arg1: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{512,12,64}{7}> {
  // expected-error @+1 {{'TFHE.add_glwe' op should have the same GLWE 'dimension' parameter}}
  %1 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<{1024,12,64}{7}>, !TFHE.glwe<{1024,12,64}{7}>) -> (!TFHE.glwe<{512,12,64}{7}>)
  return %1: !TFHE.glwe<{512,12,64}{7}>
}

// -----

// GLWE dimension parameter inputs
func.func @add_glwe(%arg0: !TFHE.glwe<{1024,12,64}{7}>, %arg1: !TFHE.glwe<{512,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}> {
  // expected-error @+1 {{'TFHE.add_glwe' op should have the same GLWE 'dimension' parameter}}
  %1 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<{1024,12,64}{7}>, !TFHE.glwe<{512,12,64}{7}>) -> (!TFHE.glwe<{1024,12,64}{7}>)
  return %1: !TFHE.glwe<{1024,12,64}{7}>
}

// -----

// GLWE polynomialSize parameter result
func.func @add_glwe(%arg0: !TFHE.glwe<{1024,12,64}{7}>, %arg1: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,11,64}{7}> {
  // expected-error @+1 {{'TFHE.add_glwe' op should have the same GLWE 'polynomialSize' parameter}}
  %1 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<{1024,12,64}{7}>, !TFHE.glwe<{1024,12,64}{7}>) -> (!TFHE.glwe<{1024,11,64}{7}>)
  return %1: !TFHE.glwe<{1024,11,64}{7}>
}

// -----

// GLWE polynomialSize parameter inputs
func.func @add_glwe(%arg0: !TFHE.glwe<{1024,12,64}{7}>, %arg1: !TFHE.glwe<{1024,11,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}> {
  // expected-error @+1 {{'TFHE.add_glwe' op should have the same GLWE 'polynomialSize' parameter}}
  %1 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<{1024,12,64}{7}>, !TFHE.glwe<{1024,11,64}{7}>) -> (!TFHE.glwe<{1024,12,64}{7}>)
  return %1: !TFHE.glwe<{1024,12,64}{7}>
}
