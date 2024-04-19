// RUN: concretecompiler --split-input-file --action=roundtrip --verify-diagnostics %s

// -----

func.func @input_output_element_type_is_different(%input: tensor<5x!FHE.eint<4>>, %indices: tensor<3xindex>) -> tensor<3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_index' op input element type '!FHE.eint<4>' doesn't match output element type '!FHE.eint<6>'}}
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<4>>, tensor<3xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

func.func @indices_last_dimension_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x4xindex>) -> tensor<3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_index' op size of the last dimension of indices '4' doesn't match the number of dimensions of input '2'}}
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<3x5x!FHE.eint<6>>, tensor<3x4xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3xindex>) -> tensor<10x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_index' op output shape '<10>' doesn't match the expected output shape '<3>'}}
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>) -> tensor<10x!FHE.eint<6>>
  return %output : tensor<10x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<2x4xindex>) -> tensor<10x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_index' op output shape '<10>' doesn't match the expected output shape '<2x4>'}}
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<2x4xindex>) -> tensor<10x!FHE.eint<6>>
  return %output : tensor<10x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x2xindex>) -> tensor<10x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_index' op output shape '<10>' doesn't match the expected output shape '<3>'}}
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<3x5x!FHE.eint<6>>, tensor<3x2xindex>) -> tensor<10x!FHE.eint<6>>
  return %output : tensor<10x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<4x6x2xindex>) -> tensor<3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_index' op output shape '<3>' doesn't match the expected output shape '<4x6>'}}
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<3x5x!FHE.eint<6>>, tensor<4x6x2xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x4x6x2xindex>) -> tensor<3x3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_index' op output shape '<3x3>' doesn't match the expected output shape '<3x4x6>'}}
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<3x5x!FHE.eint<6>>, tensor<3x4x6x2xindex>) -> tensor<3x3x!FHE.eint<6>>
  return %output : tensor<3x3x!FHE.eint<6>>
}
