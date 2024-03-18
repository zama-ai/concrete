// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// Incompatible shapes
func.func @dot_incompatible_shapes(
    %arg0: tensor<5x!FHE.eint<5>>,
    %arg1: tensor<4xi32>) -> !FHE.eint<5>
{
  // expected-error @+1 {{'FHELinalg.dot_eint_int' op arguments have incompatible shapes}}
  %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<5x!FHE.eint<5>>, tensor<4xi32>) -> !FHE.eint<5>

  return %ret : !FHE.eint<5>
}

// -----

// Incompatible input types
func.func @dot_incompatible_input_types(
    %arg0: tensor<5x!FHE.eint<2>>,
    %arg1: tensor<4xf32>) -> !FHE.eint<2>
{
  // expected-error @+1 {{'FHELinalg.dot_eint_int' op operand #1 must}}
  %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<5x!FHE.eint<2>>, tensor<4xf32>) -> !FHE.eint<2>

  return %ret : !FHE.eint<2>
}

// -----

// Wrong number of dimensions
func.func @dot_num_dims(
    %arg0: tensor<2x4x!FHE.eint<2>>,
    %arg1: tensor<2x4xi3>) -> !FHE.eint<2>
{
  // expected-error @+1 {{'FHELinalg.dot_eint_int' op operand #0}}
  %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x4x!FHE.eint<2>>, tensor<2x4xi3>) -> !FHE.eint<2>

  return %ret : !FHE.eint<2>
}

// -----

// Wrong returns type
func.func @dot_incompatible_return(
    %arg0: tensor<4x!FHE.eint<2>>,
    %arg1: tensor<4xi3>) -> !FHE.eint<3>
{
  // expected-error @+1 {{'FHELinalg.dot_eint_int' op should have the width of encrypted inputs and result equal}}
  %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<3>

  return %ret : !FHE.eint<3>
}
