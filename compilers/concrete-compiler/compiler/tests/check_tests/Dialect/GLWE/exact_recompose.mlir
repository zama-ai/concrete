// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

/////////////////////////////////////////////////
// Test exact_recompose with bodies /////////////
/////////////////////////////////////////////////

#secret_key_in = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>,
  nb_keys = <2.>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#secret_key_out = #glwe.secret_key<
  size = <dimension=<@k>, poly_size=<@n>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#encoding_in = #glwe.encoding<
  body_modulus = <2. ** 53.>,
  mask_modulus = <2. ** 53.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

#encoding_out = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

#decomposition = #glwe.decomposition<
  base = <@base>,
  level = <@level>
>

!glwe_in = !glwe.radix_glwe<
  secret_key = #secret_key_in,
  decomposition = #decomposition,
  encoding = #encoding_in
>

!glev = !glwe.glev<
  secret_key = #secret_key_out,
  encoding = #encoding_out,
  decomposition = #decomposition,
  shape = (<@K + 2.>),
  average_message = <@_secret_key.distribution.average_mean>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding_out
>

// CHECK: func.func @test_exact_recompose
func.func @test_exact_recompose_with_bodies(%arg0: !glwe_in, %glev: !glev) ->  !glwe_out {
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in, !glev) -> !glwe_out
  return %0 :  !glwe_out
}

/////////////////////////////////////////////////
// Test exact_recompose mask only ///////////////
/////////////////////////////////////////////////

#encoding_in_mask_only = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 53.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>
!glwe_in_mask_only = !glwe.radix_glwe<
  secret_key = #secret_key_in,
  decomposition = #decomposition,
  encoding = #encoding_in_mask_only,
  body = false
>

#encoding_glev = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glev_mask_only = !glwe.glev<
  secret_key = #secret_key_out,
  encoding = #encoding_glev,
  decomposition = #decomposition,
  shape = (<@K>),
  average_message = <@_secret_key.distribution.average_mean>
>

// CHECK: func.func @test_exact_recompose_mask_only
func.func @test_exact_recompose_mask_only(%arg0: !glwe_in_mask_only, %glev: !glev_mask_only) ->  !glwe_out {
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in_mask_only, !glev_mask_only) -> !glwe_out
  return %0 :  !glwe_out
}