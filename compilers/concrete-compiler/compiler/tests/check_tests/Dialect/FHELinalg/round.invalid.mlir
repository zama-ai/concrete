// RUN: concretecompiler --split-input-file --action=roundtrip --verify-diagnostics %s

// -----

func.func @mismatched_shape(%arg0: tensor<5x!FHE.eint<8>>) -> tensor<4x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.round' op input and output tensors should have the same shape}}
  %0 = "FHELinalg.round"(%arg0): (tensor<5x!FHE.eint<8>>) -> tensor<4x!FHE.eint<6>>
  return %0 : tensor<4x!FHE.eint<6>>
}

// -----

func.func @bigger_output_bit_width(%arg0: tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.eint<10>> {
  // expected-error @+1 {{'FHELinalg.round' op input tensor should have bigger bit width than output tensor}}
  %0 = "FHELinalg.round"(%arg0): (tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.eint<10>>
  return %0 : tensor<5x!FHE.eint<10>>
}

// -----

func.func @mismatched_signedness(%arg0: tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.esint<6>> {
  // expected-error @+1 {{'FHELinalg.round' op input and output tensors should have the same signedness}}
  %0 = "FHELinalg.round"(%arg0): (tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.esint<6>>
  return %0 : tensor<5x!FHE.esint<6>>
}
