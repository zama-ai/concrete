// RUN: concretecompiler --split-input-file --action=roundtrip --verify-diagnostics %s

// -----

func.func @different_input_and_output_bit_widths(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<5>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected output element type ('!FHE.eint<5>') to be the same with input element type ('!FHE.eint<7>') but it is not}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<5>>
  return %0 : tensor<1x1x13x9x!FHE.eint<5>>
}

// -----

func.func @bad_input_dimensions(%arg0: tensor<16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected input to have 4 dimensions (N*C*H*W) but it has 2}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64> } : (tensor<16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>>
  return %0 : tensor<1x1x13x9x!FHE.eint<7>>
}

// -----

func.func @bad_output_dimensions(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x13x9x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected output to have 4 dimensions (N*C*H*W) but it has 3}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x13x9x!FHE.eint<7>>
  return %0 : tensor<1x13x9x!FHE.eint<7>>
}

// -----

func.func @bad_kernel_shape_dimensions(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected kernel shape to be of shape (2) but it is of shape (3)}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2, 3]> : tensor<3xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>>
  return %0 : tensor<1x1x13x9x!FHE.eint<7>>
}

// -----

func.func @bad_strides_dimensions(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected strides to be of shape (2) but it is of shape (3)}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64>, strides = dense<[1, 1, 1]> : tensor<3xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>>
  return %0 : tensor<1x1x13x9x!FHE.eint<7>>
}

// -----

func.func @bad_strides_values(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected elements of strides to be positive but strides[0] is -1}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64>, strides = dense<[-1, 1]> : tensor<2xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>>
  return %0 : tensor<1x1x13x9x!FHE.eint<7>>
}

// -----

func.func @bad_dilations_dimensions(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected dilations to be of shape (2) but it is of shape (3)}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64>, dilations = dense<[1, 1, 1]> : tensor<3xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>>
  return %0 : tensor<1x1x13x9x!FHE.eint<7>>
}

// -----

func.func @bad_dilations_values(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected elements of dilations to be positive but dilations[1] is -1}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64>, dilations = dense<[1, -1]> : tensor<2xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>>
  return %0 : tensor<1x1x13x9x!FHE.eint<7>>
}

// -----

func.func @bad_output_shape(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x10x5x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.maxpool2d' op expected output to be of shape (1, 1, 13, 9) but it is of shape (1, 1, 10, 5)}}
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x10x5x!FHE.eint<7>>
  return %0 : tensor<1x1x10x5x!FHE.eint<7>>
}
