// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// GLWE id parameter
func.func @add_glwe_int(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[2]<12,1024>> {
  %0 = arith.constant 1 : i8
  // expected-error @+1 {{'TFHE.add_glwe_int' op should have the same GLWE Secret Key}}
  %1 = "TFHE.add_glwe_int"(%arg0, %0): (!TFHE.glwe<sk[1]<12,1024>>, i8) -> (!TFHE.glwe<sk[2]<12,1024>>)
  return %1: !TFHE.glwe<sk[2]<12,1024>>
}

// -----

// GLWE dimension parameter
func.func @add_glwe_int(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,512>> {
  %0 = arith.constant 1 : i8
  // expected-error @+1 {{'TFHE.add_glwe_int' op should have the same GLWE Secret Key}}
  %1 = "TFHE.add_glwe_int"(%arg0, %0): (!TFHE.glwe<sk[1]<12,1024>>, i8) -> (!TFHE.glwe<sk[1]<12,512>>)
  return %1: !TFHE.glwe<sk[1]<12,512>>
}

// -----

// GLWE polynomialSize parameter
func.func @add_glwe_int(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<11,1024>> {
  %0 = arith.constant 1 : i8
  // expected-error @+1 {{'TFHE.add_glwe_int' op should have the same GLWE Secret Key}}
  %1 = "TFHE.add_glwe_int"(%arg0, %0): (!TFHE.glwe<sk[1]<12,1024>>, i8) -> (!TFHE.glwe<sk[1]<11,1024>>)
  return %1: !TFHE.glwe<sk[1]<11,1024>>
}
