// RUN: concretecompiler --action=roundtrip --split-input-file --verify-diagnostics %s

// Exact decomposition should have same secret key in input and output ciphertext
#sk0 = #glwe.secret_key<
  size = <dimension=<1.>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#sk1 = #glwe.secret_key<
  size = <dimension=<2.>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#encoding = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_in = !glwe.glwe<
  secret_key = #sk0,
  encoding = #encoding
>

#decomposition = #glwe.decomposition<
  base = <@base>,
  level = <@level>
>

!glwe_out = !glwe.radix_glwe<
  secret_key = #sk1,
  encoding = #encoding,
  decomposition = #decomposition
>

func.func @test_modulus_switch(%arg0: !glwe_in) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_decompose' op failed to verify that the input and output {secret_key} parameters are equals}}
  %0 = glwe.exact_decompose %arg0 {
      decomposition = #decomposition
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}

// -----

// Exact decomposition should have same encoding in input and output ciphertext
#sk = #glwe.secret_key<
  size = <dimension=<1.>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#encoding0 = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

#encoding1 = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 5.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_in = !glwe.glwe<
  secret_key = #sk,
  encoding = #encoding0
>

#decomposition = #glwe.decomposition<
  base = <@base>,
  level = <@level>
>

!glwe_out = !glwe.radix_glwe<
  secret_key = #sk,
  encoding = #encoding1,
  decomposition = #decomposition
>

func.func @test_modulus_switch(%arg0: !glwe_in) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_decompose' op failed to verify that the input and output {encoding} parameters are equals}}
  %0 = glwe.exact_decompose %arg0 {
      decomposition = #decomposition
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}

// -----

// Exact decomposition should have same decomposition parameter than the output ciphertext
#sk = #glwe.secret_key<
  size = <dimension=<1.>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#encoding = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_in = !glwe.glwe<
  secret_key = #sk,
  encoding = #encoding
>

#decomposition0 = #glwe.decomposition<
  base = <@base>,
  level = <5.>
>

#decomposition1 = #glwe.decomposition<
  base = <@base>,
  level = <6.>
>

!glwe_out = !glwe.radix_glwe<
  secret_key = #sk,
  encoding = #encoding,
  decomposition = #decomposition1
>

func.func @test_modulus_switch(%arg0: !glwe_in) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_decompose' op failed to verify that the op and output {decomposition} parameters are equals}}
  %0 = glwe.exact_decompose %arg0 {
      decomposition = #decomposition0
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}

// -----

// Exact decomposition should have same body parameter than the output ciphertext
#sk = #glwe.secret_key<
  size = <dimension=<1.>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#encoding = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_in = !glwe.glwe<
  secret_key = #sk,
  encoding = #encoding
>

#decomposition = #glwe.decomposition<
  base = <@base>,
  level = <@level>
>

!glwe_out = !glwe.radix_glwe<
  secret_key = #sk,
  encoding = #encoding,
  decomposition = #decomposition
>

func.func @test_modulus_switch(%arg0: !glwe_in) ->  !glwe_out {
  // expected-error @+1 {{'glwe.exact_decompose' op failed to verify that the op and output {body} parameters are equals}}
  %0 = glwe.exact_decompose %arg0 {
      decomposition = #decomposition,
      body = false
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}
