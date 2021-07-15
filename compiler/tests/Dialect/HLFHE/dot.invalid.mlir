// RUN: zamacompiler --split-input-file --verify-diagnostics %s

// Unranked types
func @dot_unranked(
    %arg0: memref<?x!HLFHE.eint<2>>,
    %arg1: memref<?xi32>,
    %arg2: memref<!HLFHE.eint<2>>)
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op operand #0}}
  "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
    (memref<?x!HLFHE.eint<2>>, memref<?xi32>, memref<!HLFHE.eint<2>>) -> ()

  return
}

// -----

// Incompatible shapes
func @dot_incompatible_shapes(
    %arg0: memref<5x!HLFHE.eint<2>>,
    %arg1: memref<4xi32>,
    %arg2: memref<!HLFHE.eint<2>>)
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op arguments have incompatible shapes}}
  "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
    (memref<5x!HLFHE.eint<2>>, memref<4xi32>, memref<!HLFHE.eint<2>>) -> ()

  return
}

// -----

// Incompatible input types
func @dot_incompatible_input_types(
    %arg0: memref<4x!HLFHE.eint<2>>,
    %arg1: memref<4xf32>,
    %arg2: memref<!HLFHE.eint<2>>)
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op operand #1 must}}
  "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
    (memref<4x!HLFHE.eint<2>>, memref<4xf32>, memref<!HLFHE.eint<2>>) -> ()

  return
}

// -----

// Wrong number of dimensions
func @dot_num_dims(
    %arg0: memref<2x4x!HLFHE.eint<2>>,
    %arg1: memref<2x4xi32>,
    %arg2: memref<!HLFHE.eint<2>>)
{
  // expected-error @+1 {{'HLFHE.dot_eint_int' op operand #0}}
  "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
    (memref<2x4x!HLFHE.eint<2>>, memref<2x4xi32>, memref<!HLFHE.eint<2>>) -> ()

  return
}
