// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// GLWE dimension parameter result
func.func @add_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,512>> {
  // expected-error @+1 {{'TFHE.add_glwe' op should have the same GLWE Secret Key}}
  %1 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk[1]<12,512>>)
  return %1: !TFHE.glwe<sk[1]<12,512>>
}

// -----

// GLWE dimension parameter inputs
func.func @add_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,512>>) -> !TFHE.glwe<sk[1]<12,1024>> {
  // expected-error @+1 {{'TFHE.add_glwe' op should have the same GLWE Secret Key}}
  %1 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,512>>) -> (!TFHE.glwe<sk[1]<12,1024>>)
  return %1: !TFHE.glwe<sk[1]<12,1024>>
}

// -----

// GLWE polynomialSize parameter result
func.func @add_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<11,1024>> {
  // expected-error @+1 {{'TFHE.add_glwe' op should have the same GLWE Secret Key}}
  %1 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk[1]<11,1024>>)
  return %1: !TFHE.glwe<sk[1]<11,1024>>
}

// -----

// GLWE polynomialSize parameter inputs
func.func @add_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<11,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
  // expected-error @+1 {{'TFHE.add_glwe' op should have the same GLWE Secret Key}}
  %1 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<11,1024>>) -> (!TFHE.glwe<sk[1]<12,1024>>)
  return %1: !TFHE.glwe<sk[1]<12,1024>>
}
