// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// GLWE id parameter
func.func @neg_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[2]<12,1024>> {
  // expected-error @+1 {{'TFHE.neg_glwe' op should have the same GLWE Secret Key}}
  %1 = "TFHE.neg_glwe"(%arg0): (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk[2]<12,1024>>)
  return %1: !TFHE.glwe<sk[2]<12,1024>>
}

// -----

// GLWE dimension parameter
func.func @neg_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,512>> {
  // expected-error @+1 {{'TFHE.neg_glwe' op should have the same GLWE Secret Key}}
  %1 = "TFHE.neg_glwe"(%arg0): (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk[1]<12,512>>)
  return %1: !TFHE.glwe<sk[1]<12,512>>
}

// -----

// GLWE polynomialSize parameter
func.func @neg_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<11,1024>> {
  // expected-error @+1 {{'TFHE.neg_glwe' op should have the same GLWE Secret Key}}
  %1 = "TFHE.neg_glwe"(%arg0): (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk[1]<11,1024>>)
  return %1: !TFHE.glwe<sk[1]<11,1024>>
}
