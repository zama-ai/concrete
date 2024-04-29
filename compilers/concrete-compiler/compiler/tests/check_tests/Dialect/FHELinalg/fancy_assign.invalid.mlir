// RUN: concretecompiler --split-input-file --action=roundtrip --verify-diagnostics %s

// -----

func.func @input_values_element_type_is_different(%input: tensor<5x!FHE.eint<4>>, %indices: tensor<3xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<5x!FHE.eint<4>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op values element type '!FHE.eint<6>' doesn't match input element type '!FHE.eint<4>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x!FHE.eint<4>>, tensor<3xindex>, tensor<3x!FHE.eint<6>>) -> tensor<5x!FHE.eint<4>>
  return %output : tensor<5x!FHE.eint<4>>
}

// -----

func.func @input_output_element_type_is_different(%input: tensor<5x!FHE.eint<4>>, %indices: tensor<3xindex>, %values: tensor<3x!FHE.eint<4>>) -> tensor<5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op output element type '!FHE.eint<6>' doesn't match input element type '!FHE.eint<4>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x!FHE.eint<4>>, tensor<3xindex>, tensor<3x!FHE.eint<4>>) -> tensor<5x!FHE.eint<6>>
  return %output : tensor<5x!FHE.eint<6>>
}

// -----

func.func @indices_last_dimension_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x4xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<3x5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op size of the last dimension of indices '4' doesn't match the number of dimensions of input '2'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<3x5x!FHE.eint<6>>, tensor<3x4xindex>, tensor<3x!FHE.eint<6>>) -> tensor<3x5x!FHE.eint<6>>
  return %output : tensor<3x5x!FHE.eint<6>>
}

// -----

func.func @values_shape_is_wrong(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3xindex>, %values: tensor<2x!FHE.eint<6>>) -> tensor<5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op values shape '<2>' doesn't match the expected values shape '<3>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>, tensor<2x!FHE.eint<6>>) -> tensor<5x!FHE.eint<6>>
  return %output : tensor<5x!FHE.eint<6>>
}

// -----

func.func @values_shape_is_wrong(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3xindex>, %values: tensor<5x3x!FHE.eint<6>>) -> tensor<5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op values shape '<5x3>' doesn't match the expected values shape '<3>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>, tensor<5x3x!FHE.eint<6>>) -> tensor<5x!FHE.eint<6>>
  return %output : tensor<5x!FHE.eint<6>>
}

// -----

func.func @values_shape_is_wrong(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3x2xindex>, %values: tensor<2x3x!FHE.eint<6>>) -> tensor<5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op values shape '<2x3>' doesn't match the expected values shape '<3x2>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x!FHE.eint<6>>, tensor<3x2xindex>, tensor<2x3x!FHE.eint<6>>) -> tensor<5x!FHE.eint<6>>
  return %output : tensor<5x!FHE.eint<6>>
}

// -----

func.func @values_shape_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x2xindex>, %values: tensor<5x!FHE.eint<6>>) -> tensor<3x5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op values shape '<5>' doesn't match the expected values shape '<3>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<3x5x!FHE.eint<6>>, tensor<3x2xindex>, tensor<5x!FHE.eint<6>>) -> tensor<3x5x!FHE.eint<6>>
  return %output : tensor<3x5x!FHE.eint<6>>
}

// -----

func.func @values_shape_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x2xindex>, %values: tensor<3x2x!FHE.eint<6>>) -> tensor<3x5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op values shape '<3x2>' doesn't match the expected values shape '<3>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<3x5x!FHE.eint<6>>, tensor<3x2xindex>, tensor<3x2x!FHE.eint<6>>) -> tensor<3x5x!FHE.eint<6>>
  return %output : tensor<3x5x!FHE.eint<6>>
}

// -----

func.func @values_shape_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x4x2xindex>, %values: tensor<4x3x!FHE.eint<6>>) -> tensor<3x5x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op values shape '<4x3>' doesn't match the expected values shape '<3x4>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<3x5x!FHE.eint<6>>, tensor<3x4x2xindex>, tensor<4x3x!FHE.eint<6>>) -> tensor<3x5x!FHE.eint<6>>
  return %output : tensor<3x5x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op output shape '<3>' doesn't match the expected output shape '<5>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>, tensor<3x!FHE.eint<6>>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<5x3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op output shape '<5x3>' doesn't match the expected output shape '<5>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>, tensor<3x!FHE.eint<6>>) -> tensor<5x3x!FHE.eint<6>>
  return %output : tensor<5x3x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x2xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op output shape '<3>' doesn't match the expected output shape '<3x5>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<3x5x!FHE.eint<6>>, tensor<3x2xindex>, tensor<3x!FHE.eint<6>>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

func.func @output_shape_is_wrong(%input: tensor<3x5x!FHE.eint<6>>, %indices: tensor<3x2xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<5x3x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.fancy_assign' op output shape '<5x3>' doesn't match the expected output shape '<3x5>'}}
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<3x5x!FHE.eint<6>>, tensor<3x2xindex>, tensor<3x!FHE.eint<6>>) -> tensor<5x3x!FHE.eint<6>>
  return %output : tensor<5x3x!FHE.eint<6>>
}
