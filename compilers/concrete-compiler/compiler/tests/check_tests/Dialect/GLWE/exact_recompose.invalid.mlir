// RUN: concretecompiler --action=roundtrip --split-input-file --verify-diagnostics %s

/////////////////////////////////////////////////
// The glev size must be equal to the input secret key dimension + nb_keys if body is true
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
  shape = (<@K>),
  average_message = <@_secret_key.distribution.average_mean>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding_out
>

func.func @test_exact_recompose(%arg0: !glwe_in, %glev: !glev) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_recompose' op failed to verify that the glev size is equals to the input secret key GLWE dimension + nb_keys as body is {true}}}
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in, !glev) -> !glwe_out
  return %0 :  !glwe_out
}

// -----

/////////////////////////////////////////////////
// The glev size should be equals to the input secret key dimension as body is {false}
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
  encoding = #encoding_in,
  body = false
>

!glev = !glwe.glev<
  secret_key = #secret_key_out,
  encoding = #encoding_out,
  decomposition = #decomposition,
  shape = (<@K + 1.>),
  average_message = <@_secret_key.distribution.average_mean>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding_out
>

func.func @test_exact_recompose(%arg0: !glwe_in, %glev: !glev) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_recompose' op failed to verify that the glev size is equals to the input secret key GLWE dimension}}
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in, !glev) -> !glwe_out
  return %0 :  !glwe_out
}

// -----

/////////////////////////////////////////////////
// The glev and input GLWE ciphertext must have the same decomposition
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

#decomposition_in = #glwe.decomposition<
  base = <1.>,
  level = <@level>
>

#decomposition_glev = #glwe.decomposition<
  base = <2.>,
  level = <@level>
>

!glwe_in = !glwe.radix_glwe<
  secret_key = #secret_key_in,
  decomposition = #decomposition_in,
  encoding = #encoding_in
>

!glev = !glwe.glev<
  secret_key = #secret_key_out,
  encoding = #encoding_out,
  decomposition = #decomposition_glev,
  shape = (<@K + 2.>),
  average_message = <@_secret_key.distribution.average_mean>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding_out
>

func.func @test_exact_recompose(%arg0: !glwe_in, %glev: !glev) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_recompose' op failed to verify that the glev and input {decomposition} parameters are equals}}
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in, !glev) -> !glwe_out
  return %0 :  !glwe_out
}

// -----

/////////////////////////////////////////////////
// If input radix ciphertext is mask only, the body modulus must be equals to the glev body modulus
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

#decomposition = #glwe.decomposition<
  base = <@base>,
  level = <@level>
>

#encoding_in_mask_only = #glwe.encoding<
  body_modulus = <2. ** 53.>,
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

#encoding_out = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding_out
>

// CHECK: func.func @test_exact_recompose_mask_only
func.func @test_exact_recompose_mask_only(%arg0: !glwe_in_mask_only, %glev: !glev_mask_only) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_recompose' op failed to verify that the input and glev {encoding.body_modulus} are equals as the input {body} parameter is false}}
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in_mask_only, !glev_mask_only) -> !glwe_out
  return %0 :  !glwe_out
}

// -----

/////////////////////////////////////////////////
// The glev {body_modulus, mask_modulus} encoding parameters must be equals
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
  mask_modulus = <2. ** 63.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

#decomposition = #glwe.decomposition<
  base = <2.>,
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

func.func @test_exact_recompose(%arg0: !glwe_in, %glev: !glev) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_recompose' op failed to verify that the glev {encoding.body_modulus} is equal to glev {encoding.mask_modulus}}}
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in, !glev) -> !glwe_out
  return %0 :  !glwe_out
}

// -----

/////////////////////////////////////////////////
// The output encoding must be equal to {glev.body_modulus, glev.mask_modulus, input.message_modulus, input.right_padding, input.left_padding}
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

#encoding_glev = #glwe.encoding<
  body_modulus = <2. ** 53.>,
  mask_modulus = <2. ** 53.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

#decomposition = #glwe.decomposition<
  base = <2.>,
  level = <@level>
>

!glwe_in = !glwe.radix_glwe<
  secret_key = #secret_key_in,
  decomposition = #decomposition,
  encoding = #encoding_in
>

!glev = !glwe.glev<
  secret_key = #secret_key_out,
  encoding = #encoding_glev,
  decomposition = #decomposition,
  shape = (<@K + 2.>),
  average_message = <@_secret_key.distribution.average_mean>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding_out
>

func.func @test_exact_recompose(%arg0: !glwe_in, %glev: !glev) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_recompose' op failed to verify that output encoding match the expected encoding. The output encoding must be equal to the input encoding with {mask_modulus, body_modulus} parameter equal to the {mask_modulus} of the glev encoding}}
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in, !glev) -> !glwe_out
  return %0 :  !glwe_out
}

// -----

/////////////////////////////////////////////////
// The glev and output {secret_key} parameters must be equal
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
  base = <2.>,
  level = <@level>
>

!glwe_in = !glwe.radix_glwe<
  secret_key = #secret_key_in,
  decomposition = #decomposition,
  encoding = #encoding_in
>

!glev = !glwe.glev<
  secret_key = #secret_key_in,
  encoding = #encoding_out,
  decomposition = #decomposition,
  shape = (<@K + 2.>),
  average_message = <@_secret_key.distribution.average_mean>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding_out
>

func.func @test_exact_recompose(%arg0: !glwe_in, %glev: !glev) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_recompose' op failed to verify that the glev and output {secret_key} parameters are equals}}
  %0 = glwe.exact_recompose %arg0, %glev : (!glwe_in, !glev) -> !glwe_out
  return %0 :  !glwe_out
}