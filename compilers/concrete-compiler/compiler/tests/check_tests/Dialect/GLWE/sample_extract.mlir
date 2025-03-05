// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

// The input secret key
#secret_key_in = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>
>

// The ouput secret key must have poly_size equal to 1 and dimension equal to input secret key dimension * poly_size
#secret_key_out = #glwe.secret_key<
  size = <dimension=<@K * @N>, poly_size=<1.>>
>

// The input and output encoding must have the same parameters
#encoding = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

// Let's define the input and output GLWE types
!glwe_in = !glwe.glwe<
  secret_key = #secret_key_in,
  encoding = #encoding
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding
>

// Test sample_extract
// CHECK:  func @test_sample_extract
func.func @test_sample_extract(%arg0: !glwe_in) ->  !glwe_out {
  %0 = glwe.sample_extract %arg0 {
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}
