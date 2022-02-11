// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

func @zero_1D_scalar() -> tensor<4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHE.zero_tensor' op}}
  %0 = "FHE.zero_tensor"() : () -> !FHE.eint<2>
  return %0 : !FHE.eint<2>
}

// -----

func @zero_plaintext() -> tensor<4x9xi32> {
  // expected-error @+1 {{'FHE.zero_tensor' op}}
  %0 = "FHE.zero_tensor"() : () -> tensor<4x9xi32>
  return %0 : tensor<4x9xi32>
}