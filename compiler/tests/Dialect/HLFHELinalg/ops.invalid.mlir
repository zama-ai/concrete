// RUN: zamacompiler --split-input-file --verify-diagnostics --action=roundtrip %s

/////////////////////////////////////////////////
// HLFHELinalg.add_eint_int
/////////////////////////////////////////////////

// Incompatible dimension of operands
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.add_eint_int' op has the dimension #2 of the operand #1 incompatible with other operands, got 2 expect 1 or 3}}
  %1 = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x2x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible dimension of result
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x10x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.add_eint_int' op has the dimension #3 of the result incompatible with operands dimension, got 10 expect 2}}
  %1 = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x10x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x10x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible number of dimension between operands and result
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.add_eint_int' op should have the number of dimensions of the result equal to the highest number of dimensions of operands, got 3 expect 4}}
  %1 = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible width between clear and encrypted witdh
func @main(%a0: tensor<2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x3x4xi4>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.add_eint_int' op should have the width of integer values less or equals than the width of encrypted values + 1}}
  %1 = "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x3x4x!HLFHE.eint<2>>, tensor<2x3x4xi4>) -> tensor<2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// HLFHELinalg.add_eint
/////////////////////////////////////////////////

// Incompatible dimension of operands
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4x!HLFHE.eint<2>>) -> tensor<2x2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.add_eint' op has the dimension #2 of the operand #1 incompatible with other operands, got 2 expect 1 or 3}}
  %1 = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4x!HLFHE.eint<2>>) -> tensor<2x2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x2x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible dimension of result
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4x!HLFHE.eint<2>>) -> tensor<2x10x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.add_eint' op has the dimension #3 of the result incompatible with operands dimension, got 10 expect 2}}
  %1 = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4x!HLFHE.eint<2>>) -> tensor<2x10x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x10x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible number of dimension between operands and result
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4x!HLFHE.eint<2>>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.add_eint' op should have the number of dimensions of the result equal to the highest number of dimensions of operands, got 3 expect 4}}
  %1 = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4x!HLFHE.eint<2>>) -> tensor<2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible width between clear and encrypted witdh
func @main(%a0: tensor<2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x3x4x!HLFHE.eint<3>>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.add_eint' op should have the width of encrypted equals, got 3 expect 2}}
  %1 = "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<2x3x4x!HLFHE.eint<2>>, tensor<2x3x4x!HLFHE.eint<3>>) -> tensor<2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// HLFHELinalg.mul_eint_int
/////////////////////////////////////////////////

// Incompatible dimension of operands
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.mul_eint_int' op has the dimension #2 of the operand #1 incompatible with other operands, got 2 expect 1 or 3}}
  %1 = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x2x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible dimension of result
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x10x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.mul_eint_int' op has the dimension #3 of the result incompatible with operands dimension, got 10 expect 2}}
  %1 = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x10x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x10x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible number of dimension between operands and result
func @main(%a0: tensor<2x2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.mul_eint_int' op should have the number of dimensions of the result equal to the highest number of dimensions of operands, got 3 expect 4}}
  %1 = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!HLFHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

// Incompatible width between clear and encrypted witdh
func @main(%a0: tensor<2x3x4x!HLFHE.eint<2>>, %a1: tensor<2x3x4xi4>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.mul_eint_int' op should have the width of integer values less or equals than the width of encrypted values + 1}}
  %1 = "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<2x3x4x!HLFHE.eint<2>>, tensor<2x3x4xi4>) -> tensor<2x3x4x!HLFHE.eint<2>>
  return %1 : tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// HLFHELinalg.apply_lookup_table
/////////////////////////////////////////////////

func @apply_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<4xi32>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.apply_lookup_table' op should have as operand #2 a tensor<2^pxi64>, where p is the width of the encrypted integer of the operand #1,expect tensor <4xi64>}}
  %1 = "HLFHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!HLFHE.eint<2>>, tensor<4xi32>) -> (tensor<2x3x4x!HLFHE.eint<2>>)
  return %1: tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

func @apply_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<12xi64>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.apply_lookup_table' op should have as operand #2 a tensor<2^pxi64>, where p is the width of the encrypted integer of the operand #1,expect tensor <4xi64>}}
  %1 = "HLFHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!HLFHE.eint<2>>, tensor<12xi64>) -> (tensor<2x3x4x!HLFHE.eint<2>>)
  return %1: tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

func @apply_lookup_table(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.apply_lookup_table' op  should have same shapes for operand #1 and the result}}
  %1 = "HLFHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<3x4x!HLFHE.eint<2>>, tensor<4xi64>) -> (tensor<2x3x4x!HLFHE.eint<2>>)
  return %1: tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// HLFHELinalg.apply_multi_lookup_table
/////////////////////////////////////////////////

func @apply_multi_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<2x6xi64>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.apply_multi_lookup_table' op should have as operand #2 a tensor<DMx...xD1X2^pxi64>, where p is the width of the encrypted integer of the operand #1,expect tensor <DMx...xD1X4xi64>}}
  %1 = "HLFHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!HLFHE.eint<2>>, tensor<2x6xi64>) -> (tensor<2x3x4x!HLFHE.eint<2>>)
  return %1: tensor<2x3x4x!HLFHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// HLFHELinalg.matmul_eint_int
/////////////////////////////////////////////////

func @matmul_eint_int(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.matmul_eint_int' op should have 2D tensors as operands}}
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<2x3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

// -----

func @matmul_eint_int(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<2x4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.matmul_eint_int' op should have 2D tensors as operands}}
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x4x!HLFHE.eint<2>>, tensor<2x4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

// -----

func @matmul_eint_int(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<5x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.matmul_eint_int' op should have the dimension #0 of operand #1equals to the dimension #1 of operand #0, expect 4 got 5}}
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x4x!HLFHE.eint<2>>, tensor<5x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}

// -----

func @matmul_eint_int(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<4x2x!HLFHE.eint<2>> {
  // expected-error @+1 {{'HLFHELinalg.matmul_eint_int' op should have the result shape compatible with operands shape, expect 3x2 as the shape of the result}}
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) -> tensor<4x2x!HLFHE.eint<2>>
  return %1 : tensor<4x2x!HLFHE.eint<2>>
}


