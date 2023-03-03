// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

func.func @zero_1D_scalar() -> tensor<4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHE.zero_tensor' op}}
  %0 = "FHE.zero_tensor"() : () -> !FHE.eint<2>
  return %0 : !FHE.eint<2>
}

// -----

func.func @zero_plaintext() -> tensor<4x9xi32> {
  // expected-error @+1 {{'FHE.zero_tensor' op}}
  %0 = "FHE.zero_tensor"() : () -> tensor<4x9xi32>
  return %0 : tensor<4x9xi32>
}

// -----

func.func @to_bool(%arg0: !FHE.eint<3>) -> !FHE.ebool {
  // expected-error @+1 {{'FHE.to_bool' op}}
  %1 = "FHE.to_bool"(%arg0): (!FHE.eint<3>) -> (!FHE.ebool)
  return %1: !FHE.ebool
}

// -----

func.func @gen_gate(%arg0: !FHE.ebool, %arg1: !FHE.ebool, %arg2: tensor<5xi1>) -> !FHE.ebool {
  // expected-error @+1 {{'FHE.gen_gate' op}}
  %1 = "FHE.gen_gate"(%arg0, %arg1, %arg2) : (!FHE.ebool, !FHE.ebool, tensor<5xi1>) -> !FHE.ebool
  return %1: !FHE.ebool
}

// -----

func.func @gen_gate(%arg0: !FHE.ebool, %arg1: !FHE.ebool, %arg2: tensor<6xi64>) -> !FHE.ebool {
  // expected-error @+1 {{'FHE.gen_gate' op}}
  %1 = "FHE.gen_gate"(%arg0, %arg1, %arg2) : (!FHE.ebool, !FHE.ebool, tensor<6xi64>) -> !FHE.ebool
  return %1: !FHE.ebool
}
