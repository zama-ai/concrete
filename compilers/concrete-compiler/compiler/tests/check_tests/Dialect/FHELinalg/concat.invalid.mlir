// RUN: concretecompiler --split-input-file --action=roundtrip --verify-diagnostics %s

// -----

func.func @main() -> tensor<0x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op should have at least 1 input}}
  %0 = "FHELinalg.concat"() : () -> tensor<0x!FHE.eint<7>>
  return %0 : tensor<0x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<4x!FHE.eint<7>>, %y: tensor<3x!FHE.eint<7>>) -> tensor<7x!FHE.eint<6>> {
  // expected-error @+1 {{'FHELinalg.concat' op input element type ('!FHE.eint<7>') doesn't match output element type ('!FHE.eint<6>')}}
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<4x!FHE.eint<7>>, tensor<3x!FHE.eint<7>>) -> tensor<7x!FHE.eint<6>>
  return %0 : tensor<7x!FHE.eint<6>>
}

// -----

func.func @main(%x: tensor<4x!FHE.eint<6>>, %y: tensor<3x!FHE.eint<6>>) -> tensor<7x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op input element type ('!FHE.eint<6>') doesn't match output element type ('!FHE.eint<7>')}}
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<4x!FHE.eint<6>>, tensor<3x!FHE.eint<6>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<4x!FHE.eint<6>>, %y: tensor<3x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op input element type ('!FHE.eint<6>') doesn't match output element type ('!FHE.eint<7>')}}
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<4x!FHE.eint<6>>, tensor<3x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<4x!FHE.eint<7>>, %y: tensor<3x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op has invalid axis attribute}}
  %0 = "FHELinalg.concat"(%x, %y) { axis = 3 } : (tensor<4x!FHE.eint<7>>, tensor<3x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<4x!FHE.eint<7>>, %y: tensor<3x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op has invalid axis attribute}}
  %0 = "FHELinalg.concat"(%x, %y) { axis = -3 } : (tensor<4x!FHE.eint<7>>, tensor<3x!FHE.eint<7>>) -> tensor<7x!FHE.eint<7>>
  return %0 : tensor<7x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<4x!FHE.eint<7>>, %y: tensor<3x!FHE.eint<7>>) -> tensor<10x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op does not have the proper output shape of <7>}}
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<4x!FHE.eint<7>>, tensor<3x!FHE.eint<7>>) -> tensor<10x!FHE.eint<7>>
  return %0 : tensor<10x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<5x4x!FHE.eint<7>>) -> tensor<10x4x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op does not have the proper output shape of <8x4>}}
  %0 = "FHELinalg.concat"(%x, %y) : (tensor<3x4x!FHE.eint<7>>, tensor<5x4x!FHE.eint<7>>) -> tensor<10x4x!FHE.eint<7>>
  return %0 : tensor<10x4x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<3x5x!FHE.eint<7>>) -> tensor<3x10x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op does not have the proper output shape of <3x9>}}
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<3x4x!FHE.eint<7>>, tensor<3x5x!FHE.eint<7>>) -> tensor<3x10x!FHE.eint<7>>
  return %0 : tensor<3x10x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<3x5x!FHE.eint<7>>) -> tensor<3x4x10x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op does not have the proper shape of <?x4x10> for input #0}}
  %0 = "FHELinalg.concat"(%x, %y) { axis = 0 } : (tensor<3x4x!FHE.eint<7>>, tensor<3x5x!FHE.eint<7>>) -> tensor<3x4x10x!FHE.eint<7>>
  return %0 : tensor<3x4x10x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<3x5x!FHE.eint<7>>) -> tensor<3x4x10x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op does not have the proper shape of <3x?x10> for input #0}}
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<3x4x!FHE.eint<7>>, tensor<3x5x!FHE.eint<7>>) -> tensor<3x4x10x!FHE.eint<7>>
  return %0 : tensor<3x4x10x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<3x5x!FHE.eint<7>>) -> tensor<3x4x10x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op does not have the proper shape of <3x4x?> for input #0}}
  %0 = "FHELinalg.concat"(%x, %y) { axis = 2 } : (tensor<3x4x!FHE.eint<7>>, tensor<3x5x!FHE.eint<7>>) -> tensor<3x4x10x!FHE.eint<7>>
  return %0 : tensor<3x4x10x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<3x4x4x!FHE.eint<7>>, %y: tensor<3x5x!FHE.eint<7>>) -> tensor<3x4x10x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op does not have the proper shape of <3x4x?> for input #1}}
  %0 = "FHELinalg.concat"(%x, %y) { axis = 2 } : (tensor<3x4x4x!FHE.eint<7>>, tensor<3x5x!FHE.eint<7>>) -> tensor<3x4x10x!FHE.eint<7>>
  return %0 : tensor<3x4x10x!FHE.eint<7>>
}

// -----

func.func @main(%x: tensor<3x4x4x!FHE.eint<7>>, %y: tensor<3x5x4x!FHE.eint<7>>) -> tensor<3x10x4x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.concat' op does not have the proper output shape of <3x9x4>}}
  %0 = "FHELinalg.concat"(%x, %y) { axis = 1 } : (tensor<3x4x4x!FHE.eint<7>>, tensor<3x5x4x!FHE.eint<7>>) -> tensor<3x10x4x!FHE.eint<7>>
  return %0 : tensor<3x10x4x!FHE.eint<7>>
}
