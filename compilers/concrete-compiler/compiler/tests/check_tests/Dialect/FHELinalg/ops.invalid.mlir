// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

/////////////////////////////////////////////////
// FHELinalg.add_eint_int
/////////////////////////////////////////////////

// Incompatible dimension of operands
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.add_eint_int' op has the dimension #2 of the operand #1 incompatible with other operands, got 2 expect 1 or 3}}
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x2x3x4x!FHE.eint<2>>
}

// -----

// Incompatible dimension of result
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x10x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.add_eint_int' op has the dimension #3 of the result incompatible with operands dimension, got 10 expect 2}}
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x10x3x4x!FHE.eint<2>>
  return %1 : tensor<2x10x3x4x!FHE.eint<2>>
}

// -----

// Incompatible number of dimension between operands and result
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.add_eint_int' op should have the number of dimensions of the result equal to the highest number of dimensions of operands, got 3 expect 4}}
  %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x3x4x!FHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.add_eint
/////////////////////////////////////////////////

// Incompatible dimension of operands
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.add_eint' op has the dimension #2 of the operand #1 incompatible with other operands, got 2 expect 1 or 3}}
  %1 = "FHELinalg.add_eint"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x2x3x4x!FHE.eint<2>>
}

// -----

// Incompatible dimension of result
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x10x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.add_eint' op has the dimension #3 of the result incompatible with operands dimension, got 10 expect 2}}
  %1 = "FHELinalg.add_eint"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x10x3x4x!FHE.eint<2>>
  return %1 : tensor<2x10x3x4x!FHE.eint<2>>
}

// -----

// Incompatible number of dimension between operands and result
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.add_eint' op should have the number of dimensions of the result equal to the highest number of dimensions of operands, got 3 expect 4}}
  %1 = "FHELinalg.add_eint"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x3x4x!FHE.eint<2>>
}

// -----

// Incompatible width between clear and encrypted witdh
func.func @main(%a0: tensor<2x3x4x!FHE.eint<2>>, %a1: tensor<2x3x4x!FHE.eint<3>>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.add_eint' op should have the width of encrypted equals, got 3 expect 2}}
  %1 = "FHELinalg.add_eint"(%a0, %a1) : (tensor<2x3x4x!FHE.eint<2>>, tensor<2x3x4x!FHE.eint<3>>) -> tensor<2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x3x4x!FHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.mul_eint_int
/////////////////////////////////////////////////

// Incompatible dimension of operands
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.mul_eint_int' op has the dimension #2 of the operand #1 incompatible with other operands, got 2 expect 1 or 3}}
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x2x3x4x!FHE.eint<2>>
}

// -----

// Incompatible dimension of result
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x10x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.mul_eint_int' op has the dimension #3 of the result incompatible with operands dimension, got 10 expect 2}}
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x10x3x4x!FHE.eint<2>>
  return %1 : tensor<2x10x3x4x!FHE.eint<2>>
}

// -----

// Incompatible number of dimension between operands and result
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4xi3>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.mul_eint_int' op should have the number of dimensions of the result equal to the highest number of dimensions of operands, got 3 expect 4}}
  %1 = "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4xi3>) -> tensor<2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x3x4x!FHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.mul_eint
/////////////////////////////////////////////////

// Incompatible dimension of operands
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.mul_eint' op has the dimension #2 of the operand #1 incompatible with other operands, got 2 expect 1 or 3}}
  %1 = "FHELinalg.mul_eint"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x2x3x4x!FHE.eint<2>>
}

// -----

// Incompatible dimension of result
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x10x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.mul_eint' op has the dimension #3 of the result incompatible with operands dimension, got 10 expect 2}}
  %1 = "FHELinalg.mul_eint"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x10x3x4x!FHE.eint<2>>
  return %1 : tensor<2x10x3x4x!FHE.eint<2>>
}

// -----

// Incompatible number of dimension between operands and result
func.func @main(%a0: tensor<2x2x3x4x!FHE.eint<2>>, %a1: tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.mul_eint' op should have the number of dimensions of the result equal to the highest number of dimensions of operands, got 3 expect 4}}
  %1 = "FHELinalg.mul_eint"(%a0, %a1) : (tensor<2x2x3x4x!FHE.eint<2>>, tensor<2x2x2x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x3x4x!FHE.eint<2>>
}

// -----

// Incompatible width between clear and encrypted witdh
func.func @main(%a0: tensor<2x3x4x!FHE.eint<2>>, %a1: tensor<2x3x4x!FHE.eint<3>>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.mul_eint' op should have the width of encrypted equals, got 3 expect 2}}
  %1 = "FHELinalg.mul_eint"(%a0, %a1) : (tensor<2x3x4x!FHE.eint<2>>, tensor<2x3x4x!FHE.eint<3>>) -> tensor<2x3x4x!FHE.eint<2>>
  return %1 : tensor<2x3x4x!FHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.apply_lookup_table
/////////////////////////////////////////////////

func.func @apply_lookup_table(%arg0: tensor<2x3x4x!FHE.eint<2>>, %arg1: tensor<4xi65>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.apply_lookup_table' op should have as operand #2 a tensor<2^pxi{8,16,32,64}>, where p is the width of the encrypted integer of the operand #1,expect tensor <4xi{8,16,32,64}>}}
  %1 = "FHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!FHE.eint<2>>, tensor<4xi65>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}

// -----

func.func @apply_lookup_table(%arg0: tensor<2x3x4x!FHE.eint<2>>, %arg1: tensor<12xi64>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.apply_lookup_table' op should have as operand #2 a tensor<2^pxi{8,16,32,64}>, where p is the width of the encrypted integer of the operand #1,expect tensor <4xi{8,16,32,64}>}}
  %1 = "FHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!FHE.eint<2>>, tensor<12xi64>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}

// -----

func.func @apply_lookup_table(%arg0: tensor<3x4x!FHE.eint<2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.apply_lookup_table' op  should have same shapes for operand #1 and the result}}
  %1 = "FHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<3x4x!FHE.eint<2>>, tensor<4xi64>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.apply_multi_lookup_table
/////////////////////////////////////////////////

func.func @apply_multi_lookup_table(%arg0: tensor<2x3x4x!FHE.eint<2>>, %arg1: tensor<2x6xi64>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.apply_multi_lookup_table' op should have as operand #2 a tensor<DMx...xD1X2^pxi{8,16,32,64}>, where p is the width of the encrypted integer of the operand #1,expect tensor <DMx...xD1X4xi{8,16,32,64}>}}
  %1 = "FHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!FHE.eint<2>>, tensor<2x6xi64>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}

// -----

func.func @apply_multi_lookup_table_bad_prec(%arg0: tensor<2x3x4x!FHE.eint<2>>, %arg1: tensor<2x4xi65>) -> tensor<2x3x4x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.apply_multi_lookup_table' op should have as operand #2 a tensor<DMx...xD1X2^pxi{8,16,32,64}>, where p is the width of the encrypted integer of the operand #1,expect tensor <DMx...xD1X4xi{8,16,32,64}>}}
  %1 = "FHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!FHE.eint<2>>, tensor<2x4xi65>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}

// -----


/////////////////////////////////////////////////
// FHELinalg.apply_mapped_lookup_table
/////////////////////////////////////////////////
func.func @apply_mapped_lookup_table_bad_lut_size_127_vs_128(
  %input: tensor<2x3x4x!FHE.eint<7>>,
  %luts: tensor<127xi64>,
  %map: tensor<2x3x4xindex>
) -> tensor<2x3x4x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.apply_mapped_lookup_table' op : `luts` (operand #2) inner dimension should have size 128(=2^7) to match `ct` (operand #1) elements bitwidth (7)}}
  %1 = "FHELinalg.apply_mapped_lookup_table"(%input, %luts, %map): (tensor<2x3x4x!FHE.eint<7>>, tensor<127xi64>, tensor<2x3x4xindex>) -> tensor<2x3x4x!FHE.eint<7>>
  return %1: tensor<2x3x4x!FHE.eint<7>>
}

// -----

func.func @apply_mapped_lookup_table_bad_map_size(
  %input: tensor<2x3x4x!FHE.eint<7>>,
  %luts: tensor<128xi64>,
  %map: tensor<2x3xindex>
) -> tensor<2x3x4x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.apply_mapped_lookup_table' op : 't' (operand #1) rank (=3) differs from 'lut_map.getName()' (operand #3) rank (=2)}}
  %1 = "FHELinalg.apply_mapped_lookup_table"(%input, %luts, %map): (tensor<2x3x4x!FHE.eint<7>>, tensor<128xi64>, tensor<2x3xindex>) -> tensor<2x3x4x!FHE.eint<7>>
  return %1: tensor<2x3x4x!FHE.eint<7>>
}

// -----

func.func @apply_mapped_lookup_table_bad_map_elmt_type(
  %input: tensor<2x3x4x!FHE.eint<7>>,
  %luts: tensor<128xi64>,
  %map: tensor<2x3xindex>
) -> tensor<2x3x4x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.apply_mapped_lookup_table' op : 't' (operand #1) rank (=3) differs from 'lut_map.getName()' (operand #3) rank (=2)}}
  %1 = "FHELinalg.apply_mapped_lookup_table"(%input, %luts, %map): (tensor<2x3x4x!FHE.eint<7>>, tensor<128xi64>, tensor<2x3xindex>) -> tensor<2x3x4x!FHE.eint<7>>
  return %1: tensor<2x3x4x!FHE.eint<7>>
}

// -----

func.func @apply_mapped_lookup_table_bad_lut_prec(
  %input: tensor<2x3x4x!FHE.eint<7>>,
  %luts: tensor<128xi65>,
  %map: tensor<2x3x4xindex>
) -> tensor<2x3x4x!FHE.eint<7>> {
  // expected-error @+1 {{'FHELinalg.apply_mapped_lookup_table' op should have as operand #2 a tensor<DMx...xD1X2^pxi{8,16,32,64}>, where p is the width of the encrypted integer of the operand #1,expect tensor <DMx...xD1X128xi{8,16,32,64}>}}
  %0 = "FHELinalg.apply_mapped_lookup_table"(%input, %luts, %map): (tensor<2x3x4x!FHE.eint<7>>, tensor<128xi65>, tensor<2x3x4xindex>) -> (tensor<2x3x4x!FHE.eint<7>>)
  return %0: tensor<2x3x4x!FHE.eint<7>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.conv2d
/////////////////////////////////////////////////

func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op padding isn't yet supported, but got a non zero value (1) at index 0}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[1,0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}

// -----


func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op group must be strictly positif, but got 0}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){group = 0 : i64}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}

// -----


func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected number of channels in weight to be equal to 1 (input_channels / group) but got 3}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){group = 3 : i64}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}

// -----


func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x1x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected number of feature maps (4) to be a multiple of group (3)}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){group = 3 : i64}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x1x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}

// -----


func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<6x1x14x14xi3>, %bias: tensor<6xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected number of output channels to be equal to the number of filters (6) but got 4}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){group = 3 : i64}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<6x1x14x14xi3>, tensor<6xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}

// -----

func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected height of output to be equal to 8 but got 15}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[2,2]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0 ,0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}


// -----


func.func @conv2d(%input: tensor<101x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected result batch size to be equal to input batch size (101) but got 100}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0 ,0, 0, 0]> : tensor<4xi64>}: (tensor<101x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}

// -----

func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x2x2xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected height of output to be equal to 27 but got 15}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0,0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x2x2xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}


// -----

func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x2xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected width of output to be equal to 27 but got 15}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0,0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x2xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}


// -----

func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected height of output to be equal to 2 but got 15}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[2,1]> : tensor<2xi64>, padding = dense<[0,0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}

// -----

func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  // expected-error @+1 {{'FHELinalg.conv2d' op expected width of output to be equal to 2 but got 15}}
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,2]> : tensor<2xi64>, padding = dense<[0,0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}
