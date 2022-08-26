// RUN: concretecompiler --split-input-file --action=roundtrip --verify-diagnostics %s

// -----

func.func @sum_invalid_bitwidth(%arg0: tensor<4x!FHE.eint<7>>) -> !FHE.eint<6> {
  // expected-error @+1 {{'FHELinalg.sum' op should have the width of encrypted inputs and result equal}}
  %1 = "FHELinalg.sum"(%arg0): (tensor<4x!FHE.eint<7>>) -> !FHE.eint<6>
  return %1 : !FHE.eint<6>
}

// -----

func.func @sum_invalid_axes_1(%arg0: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  // expected-error @+1 {{'FHELinalg.sum' op has invalid axes attribute}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [4] } : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %1 : !FHE.eint<7>
}

// -----

func.func @sum_invalid_axes_2(%arg0: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  // expected-error @+1 {{'FHELinalg.sum' op has invalid axes attribute}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [-1] } : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %1 : !FHE.eint<7>
}

// -----

func.func @sum_invalid_shape_01(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <>}}
  %1 = "FHELinalg.sum"(%arg0) : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_02(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <3x4x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_03(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x4x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [1] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_04(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x3x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_05(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x3x4>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_06(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <4x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 1] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_07(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <3x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_08(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <3x4>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_09(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [1, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_10(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x4>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [1, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_11(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x3>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_12(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_13(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <4>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_14(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <3>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_15(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [1, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_16(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_17(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x1x1x1>}}
  %1 = "FHELinalg.sum"(%arg0) { keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_18(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x3x4x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_19(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x1x4x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [1], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_20(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x3x1x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_21(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x3x4x1>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_22(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x1x4x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 1], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_23(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x3x1x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_24(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x3x4x1>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_25(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x1x1x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [1, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_26(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x1x4x1>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [1, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_27(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x3x1x1>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_28(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x1x1x2>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_29(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x1x4x1>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_30(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x3x1x1>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_31(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <5x1x1x1>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [1, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}

// -----

func.func @sum_invalid_shape_32(%arg0: tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.sum' op does not have the proper output shape of <1x1x1x1>}}
  %1 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<10x20x!FHE.eint<7>>
  return %1 : tensor<10x20x!FHE.eint<7>>
}
