// RUN: concretecompiler --split-input-file --action=roundtrip --verify-diagnostics %s

// -----

func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<2x3xi3>) -> tensor<4x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op should have the same size on dimension #1 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<2x3xi3>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<2x3x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op should have the same size on dimension #1 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<2x3x!FHE.eint<2>>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x3xi3>, %y: tensor<4x3x!FHE.eint<2>>) -> tensor<2x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op should have the same size on dimension #1 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x3xi3>, tensor<4x3x!FHE.eint<2>>) -> tensor<2x4x!FHE.eint<2>>
  return %0 : tensor<2x4x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x5x!FHE.eint<2>>, %y: tensor<4x3x2xi3>) -> tensor<2x4x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op should have the same size on dimension #3 of operand #0 and dimension #1 of operand #1}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x5x!FHE.eint<2>>, tensor<4x3x2xi3>) -> tensor<2x4x3x2x!FHE.eint<2>>
  return %0 : tensor<2x4x3x2x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x4x3x5x!FHE.eint<2>>, %y: tensor<4x3x2x!FHE.eint<2>>) -> tensor<2x4x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op should have the same size on dimension #3 of operand #0 and dimension #1 of operand #1}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x4x3x5x!FHE.eint<2>>, tensor<4x3x2x!FHE.eint<2>>) -> tensor<2x4x3x2x!FHE.eint<2>>
  return %0 : tensor<2x4x3x2x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x5xi3>, %y: tensor<4x3x2x!FHE.eint<2>>) -> tensor<2x4x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op should have the same size on dimension #3 of operand #0 and dimension #1 of operand #1}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3x5xi3>, tensor<4x3x2x!FHE.eint<2>>) -> tensor<2x4x3x2x!FHE.eint<2>>
  return %0 : tensor<2x4x3x2x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x5x!FHE.eint<2>>, %y: tensor<10x5x2xi3>) -> tensor<2x4x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op should have the same size or size of 1 on dimension #1 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x5x!FHE.eint<2>>, tensor<10x5x2xi3>) -> tensor<2x4x3x2x!FHE.eint<2>>
  return %0 : tensor<2x4x3x2x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x4x3x5x!FHE.eint<2>>, %y: tensor<10x5x2x!FHE.eint<2>>) -> tensor<2x4x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op should have the same size or size of 1 on dimension #1 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x4x3x5x!FHE.eint<2>>, tensor<10x5x2x!FHE.eint<2>>) -> tensor<2x4x3x2x!FHE.eint<2>>
  return %0 : tensor<2x4x3x2x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x5xi3>, %y: tensor<10x5x2x!FHE.eint<2>>) -> tensor<2x4x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op should have the same size or size of 1 on dimension #1 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3x5xi3>, tensor<10x5x2x!FHE.eint<2>>) -> tensor<2x4x3x2x!FHE.eint<2>>
  return %0 : tensor<2x4x3x2x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x!FHE.eint<2>>, %y: tensor<5x3x4x2xi3>) -> tensor<5x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op should have the same size on dimension #0 of operand #0 and dimension #2 of operand #1}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x!FHE.eint<2>>, tensor<5x3x4x2xi3>) -> tensor<5x3x2x!FHE.eint<2>>
  return %0 : tensor<5x3x2x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x!FHE.eint<2>>, %y: tensor<5x3x4x2x!FHE.eint<2>>) -> tensor<5x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op should have the same size on dimension #0 of operand #0 and dimension #2 of operand #1}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x!FHE.eint<2>>, tensor<5x3x4x2x!FHE.eint<2>>) -> tensor<5x3x2x!FHE.eint<2>>
  return %0 : tensor<5x3x2x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2xi3>, %y: tensor<5x3x4x2x!FHE.eint<2>>) -> tensor<5x3x2x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op should have the same size on dimension #0 of operand #0 and dimension #2 of operand #1}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2xi3>, tensor<5x3x4x2x!FHE.eint<2>>) -> tensor<5x3x2x!FHE.eint<2>>
  return %0 : tensor<5x3x2x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x3x4x2x!FHE.eint<2>>, %y: tensor<4xi3>) -> tensor<5x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op should have the same size on dimension #3 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x3x4x2x!FHE.eint<2>>, tensor<4xi3>) -> tensor<5x3x4x!FHE.eint<2>>
  return %0 : tensor<5x3x4x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<5x3x4x2x!FHE.eint<2>>, %y: tensor<4x!FHE.eint<2>>) -> tensor<5x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op should have the same size on dimension #3 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<5x3x4x2x!FHE.eint<2>>, tensor<4x!FHE.eint<2>>) -> tensor<5x3x4x!FHE.eint<2>>
  return %0 : tensor<5x3x4x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x3x4x2xi3>, %y: tensor<4x!FHE.eint<2>>) -> tensor<5x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op should have the same size on dimension #3 of operand #0 and dimension #0 of operand #1}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x3x4x2xi3>, tensor<4x!FHE.eint<2>>) -> tensor<5x3x4x!FHE.eint<2>>
  return %0 : tensor<5x3x4x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4x!FHE.eint<2>>, %y: tensor<4xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op should have at least one multi dimensional tensor as an operand}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<4x!FHE.eint<2>>, %y: tensor<4x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op should have at least one multi dimensional tensor as an operand}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<4x!FHE.eint<2>>, tensor<4x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4xi3>, %y: tensor<4x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op should have at least one multi dimensional tensor as an operand}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4xi3>, tensor<4x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4x3xi3>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<3xi3>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<4x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<4x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<4x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<4x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<3xi3>, %y: tensor<4x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<3xi3>, tensor<4x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<4x5x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <4x5x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<4x5x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<3x!FHE.eint<2>>, %y: tensor<4x5x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <4x5x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<3x!FHE.eint<2>>, tensor<4x5x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<3xi3>, %y: tensor<4x5x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <4x5x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<3xi3>, tensor<4x5x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<3xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <4>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <4>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4x3xi3>, %y: tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <4>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<3xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <2x4>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <2x4>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <2x4>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<3xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <5x2x4>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<3xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <5x2x4>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x2x4x3xi3>, %y: tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <5x2x4>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x2x4x3xi3>, tensor<3x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<1x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<1x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<1x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<1x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<1x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<1x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<1x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<1x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<1x4x3x!FHE.eint<2>>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<1x4x3x!FHE.eint<2>>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<1x4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<1x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x2x4x3xi3>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x2x4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x2x4x3xi3>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x2x4x3xi3>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<5x2x4x3x!FHE.eint<2>>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<5x2x4x3x!FHE.eint<2>>, tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x2x4x3xi3>, %y: tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<5x2x4x3xi3>, tensor<3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<2x4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<2x4x3x!FHE.eint<2>>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<2x4x3xi3>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<2x4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<5x2x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<4x3x!FHE.eint<2>>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<4x3x!FHE.eint<2>>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<4x3xi3>, %y: tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_int_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<4x3xi3>, tensor<5x2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

func.func @main(%x: tensor<5x1x4x3x!FHE.eint<2>>, %y: tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_int' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x1x4x3x!FHE.eint<2>>, tensor<2x3x2xi3>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}


// -----


func.func @main(%x: tensor<5x1x4x3x!FHE.eint<2>>, %y: tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.matmul_eint_eint' op does not have the proper output shape of <5x2x4x2>}}
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<5x1x4x3x!FHE.eint<2>>, tensor<2x3x2x!FHE.eint<2>>) -> tensor<1x!FHE.eint<2>>
  return %0 : tensor<1x!FHE.eint<2>>
}

// -----

