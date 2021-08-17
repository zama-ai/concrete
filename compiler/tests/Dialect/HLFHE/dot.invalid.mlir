// RUN: zamacompiler --split-input-file --verify-diagnostics %s

// Incompatible shapes
func @dot_incompatible_shapes(
    %arg0: tensor<5x!HLFHE.eint<5>>,
    %arg1: tensor<4xi32>) -> !HLFHE.eint<5>
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op arguments have incompatible shapes}}
  %ret = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<5x!HLFHE.eint<5>>, tensor<4xi32>) -> !HLFHE.eint<5>

  return %ret : !HLFHE.eint<5>
}

// -----

// Incompatible input types
func @dot_incompatible_input_types(
    %arg0: tensor<5x!HLFHE.eint<2>>,
    %arg1: tensor<4xf32>) -> !HLFHE.eint<2>
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op operand #1 must}}
  %ret = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<5x!HLFHE.eint<2>>, tensor<4xf32>) -> !HLFHE.eint<2>

  return %ret : !HLFHE.eint<2>
}

// -----

// Wrong number of dimensions
func @dot_num_dims(
    %arg0: tensor<2x4x!HLFHE.eint<2>>,
    %arg1: tensor<2x4xi3>) -> !HLFHE.eint<2>
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op operand #0}}
  %ret = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x4x!HLFHE.eint<2>>, tensor<2x4xi3>) -> !HLFHE.eint<2>

  return %ret : !HLFHE.eint<2>
}

// -----

// Wrong returns type
func @dot_incompatible_return(
    %arg0: tensor<4x!HLFHE.eint<2>>,
    %arg1: tensor<4xi3>) -> !HLFHE.eint<3>
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op  should have the width of encrypted inputs and result equals}}
  %ret = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<4x!HLFHE.eint<2>>, tensor<4xi3>) -> !HLFHE.eint<3>

  return %ret : !HLFHE.eint<3>
}

// -----

// Wrong integer size
func @dot_incompatible_int(
    %arg0: tensor<4x!HLFHE.eint<2>>,
    %arg1: tensor<4xi4>) -> !HLFHE.eint<2>
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op  should have the width of plain input equals to width of encrypted input + 1}}
  %ret = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<4x!HLFHE.eint<2>>, tensor<4xi4>) -> !HLFHE.eint<2>

  return %ret : !HLFHE.eint<2>
}