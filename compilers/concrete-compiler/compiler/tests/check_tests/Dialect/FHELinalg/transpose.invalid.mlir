// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// Incompatible types
func.func @transpose_eint(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.transpose' op input and output tensors should have the same element type}}
  %c = "FHELinalg.transpose"(%arg0) : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<7>>
  return %c : tensor<5x4x3x!FHE.eint<7>>
}

// -----

// Incompatible shapes
func.func @transpose_eint(%arg0: tensor<3x4x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.transpose' op input and output tensors should have the same number of dimensions}}
  %c = "FHELinalg.transpose"(%arg0) : (tensor<3x4x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>>
  return %c : tensor<5x4x3x!FHE.eint<6>>
}

// -----

// Incompatible shapes
func.func @transpose_eint(%arg0: tensor<3x4x6x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.transpose' op output tensor should have inverted dimensions of input}}
  %c = "FHELinalg.transpose"(%arg0) : (tensor<3x4x6x!FHE.eint<6>>) -> tensor<5x4x3x!FHE.eint<6>>
  return %c : tensor<5x4x3x!FHE.eint<6>>
}

// -----

func.func @transpose_eint(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.transpose' op has invalid axes attribute (doesn't have 3 elements)}}
  %c = "FHELinalg.transpose"(%arg0) { axes = [0] } : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x5x!FHE.eint<6>>
  return %c : tensor<4x3x5x!FHE.eint<6>>
}

// -----

func.func @transpose_eint(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.transpose' op has invalid axes attribute (axes[1] isn't in range [0, 2])}}
  %c = "FHELinalg.transpose"(%arg0) { axes = [1, 5, 2] } : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x5x!FHE.eint<6>>
  return %c : tensor<4x3x5x!FHE.eint<6>>
}

// -----

func.func @transpose_eint(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x10x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.transpose' op has invalid output shape (output.shape[2] is not input.shape[axes[2]])}}
  %c = "FHELinalg.transpose"(%arg0) { axes = [1, 0, 2] } : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x10x!FHE.eint<6>>
  return %c : tensor<4x3x10x!FHE.eint<6>>
}

// -----

func.func @transpose_eint(%arg0: tensor<2x2x2x!FHE.eint<6>>) -> tensor<2x2x2x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.transpose' op has invalid axes attribute (doesn't contain all input axes)}}
  %c = "FHELinalg.transpose"(%arg0) { axes = [0, 1, 0] } : (tensor<2x2x2x!FHE.eint<6>>) -> tensor<2x2x2x!FHE.eint<6>>
  return %c : tensor<2x2x2x!FHE.eint<6>>
}
