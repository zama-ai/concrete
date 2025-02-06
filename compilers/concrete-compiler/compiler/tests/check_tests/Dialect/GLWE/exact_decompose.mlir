// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

#secret_key = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#encoding = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

#decomposition = #glwe.decomposition<
  base_log = <@base_log>,
  level = <@level>
>

!glwe_in = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #encoding
>

!glwe_out = !glwe.radix_glwe<
  secret_key = #secret_key,
  encoding = #encoding,
  decomposition = #decomposition
>

// Test exact_decompose
func.func @test_exact_decompose(%arg0: !glwe_in) ->  !glwe_out {
  %0 = glwe.exact_decompose %arg0 {
      decomposition = #decomposition
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}

!glwe_out_mask_only = !glwe.radix_glwe<
  secret_key = #secret_key,
  encoding = #encoding,
  decomposition = #decomposition,
  body = false
>

// Test exact_decompose mask only
func.func @test_exact_decompose_mask_only(%arg0: !glwe_in) ->  !glwe_out_mask_only {
  %0 = glwe.exact_decompose %arg0 {
      decomposition = #decomposition,
      body = false
  } : !glwe_in ->  !glwe_out_mask_only
  return %0 :  !glwe_out_mask_only
}
