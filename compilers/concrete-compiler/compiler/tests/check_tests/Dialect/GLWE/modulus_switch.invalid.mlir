// RUN: concretecompiler --split-input-file --verify-diagnostics %s

// Modulus switch should have same secret key in inpit and output ciphertext
#sk0 = #glwe.secret_key<
  size = <dimension=<1.>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

#sk1 = #glwe.secret_key<
  size = <dimension=<2.>, poly_size=<@N>>,
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
  secret_key = #sk0,
  encoding = #input_encoding
>

!glwe_out = !glwe.glwe<
  secret_key = #sk1,
  encoding = #output_encoding
>

func.func @invalid_modulus_switch_sk_not_equals(%arg0: !glwe_in) ->  !glwe_out {
  // expected-error @+1 {{'glwe.modulus_switch' op failed to verify that the input and output {secret_key} parameters are equals}}
  %0 = glwe.modulus_switch %arg0 {
      modulus = #glwe.expr<2. ** 53.>
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}

// -----

// Modulus switch should have same modulus parameters than the ouput ciphertext
#sk = #glwe.secret_key<
  size = <dimension=<1.>, poly_size=<@N>>,
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
  secret_key = #sk,
  encoding = #input_encoding
>

!glwe_out = !glwe.glwe<
  secret_key = #sk,
  encoding = #output_encoding
>

func.func @invalid_modulus_switch_modulus_body_not_match(%arg0: !glwe_in) ->  !glwe_out {
  // expected-error @+1 {{'glwe.modulus_switch' op failed to verify that the {modulus} and output {encoding.body_modulus} parameters are equals}}
  %0 = glwe.modulus_switch %arg0 {
      modulus = #glwe.expr<2. ** 52.>
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}

// -----

// Modulus switch should have same modulus parameters than the ouput ciphertext
#sk = #glwe.secret_key<
  size = <dimension=<1.>, poly_size=<@N>>,
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
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 53.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_in = !glwe.glwe<
  secret_key = #sk,
  encoding = #input_encoding
>

!glwe_out = !glwe.glwe<
  secret_key = #sk,
  encoding = #output_encoding
>

func.func @invalid_modulus_switch_modulus_mask_not_match(%arg0: !glwe_in) ->  !glwe_out {
  // expected-error @+1 {{'glwe.modulus_switch' op failed to verify that the {modulus} and output {encoding.mask_modulus} parameters are equals}}
  %0 = glwe.modulus_switch %arg0 {
      modulus = #glwe.expr<2. ** 52.>,
      body = false
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}

// -----

// Modulus switch should have same encoding unless the one is switched 
#sk = #glwe.secret_key<
  size = <dimension=<1.>, poly_size=<@N>>,
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
  body_modulus = <2. ** 62.>,
  mask_modulus = <2. ** 53.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

!glwe_in = !glwe.glwe<
  secret_key = #sk,
  encoding = #input_encoding
>

!glwe_out = !glwe.glwe<
  secret_key = #sk,
  encoding = #output_encoding
>

func.func @invalid_modulus_switch_modulus_encoding_not_match(%arg0: !glwe_in) ->  !glwe_out {
  // expected-error @+1 {{'glwe.modulus_switch' op failed to verify that the input and output {encoding} parameters matches}}
  %0 = glwe.modulus_switch %arg0 {
      modulus = #glwe.expr<2. ** 52.>,
      body = false
  } : !glwe_in ->  !glwe_out
  return %0 :  !glwe_out
}