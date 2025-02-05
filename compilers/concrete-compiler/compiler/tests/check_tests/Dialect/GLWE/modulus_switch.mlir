// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

#secret_key = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#input_encoding = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

#output_encoding = #glwe.encoding<
  body_modulus = <2. ** 53.>,
  mask_modulus = <2. ** 53.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_in = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #input_encoding
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #output_encoding
>

// Test modulus_switch
func.func @test_modulus_switch(%arg0: !glwe_in) ->  !glwe_out {
  %0 = glwe.modulus_switch %arg0 {
      modulus = #glwe.expr<2. ** 53.>
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}

// Test modulus_switch with mask only
#output_encoding_mask_only = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 53.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_mask_only = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #output_encoding_mask_only
>
func.func @test_modulus_switch_mask_only(%arg0: !glwe_in) ->  !glwe_mask_only {
  %0 = glwe.modulus_switch %arg0 {
      modulus = #glwe.expr<2. ** 53.>,
      body = false
  } : !glwe_in ->  !glwe_mask_only
  return %0 :  !glwe_mask_only
}