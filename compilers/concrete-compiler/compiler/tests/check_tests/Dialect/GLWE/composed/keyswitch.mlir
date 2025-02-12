// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

// Test a specification of a keysicth

// Define the decomposition with free variables @b and @l,
// respectivly for base_log and level parameters of the decomposition 
#decomposition = #glwe.decomposition<
  base_log = <@b>,
  level = <@l>
>

// Define the keyswitch as @b power of @l
#keyswitch_modulus = #glwe.expr<@b ** @l>

// Define the input secret key with a fixed distribution (gaussian)
// and free variables @K and @N respectively for the glwe dimesion and the size of the polynomial.
#secret_key_in = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

// Define the input encoding of the GLWE ciphertext, here it's a 6bits native encoding.
#encoding_in = #glwe.encoding<
  body_modulus = <2. ** 64.>,
  mask_modulus = <2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

// Define the GLWE input ciphertext type
!glwe_in = !glwe.glwe<
  secret_key = #secret_key_in,
  encoding = #encoding_in
>

// Define the encoding of the GLWE switched ciphertext (only mask is switched)
#encoding_switched = #glwe.encoding<
  body_modulus = < 2. ** 64.>,
  mask_modulus = #keyswitch_modulus,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

// Define the type of the GLWE switched ciphertext
!glwe_switched = !glwe.glwe<
  secret_key = #secret_key_in,
  encoding = #encoding_switched
>

// Define the type of the GLWE decomposed ciphertext
!glwe_decomposed = !glwe.radix_glwe<
  secret_key = #secret_key_in,
  encoding = #encoding_switched,
  decomposition = #decomposition,
  body = false
>

// Define the output secret key used for the recomposition
#secret_key_out = #glwe.secret_key<
  size = <dimension=<@K2>, poly_size=<@N2>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

// Define the type of glev for the recomposition
!glev = !glwe.glev<
  secret_key = #secret_key_out,
  encoding = #encoding_in,
  decomposition = #decomposition,
  shape = (<@K>),
  average_message = <@_secret_key.distribution.average_mean>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key_out,
  encoding = #encoding_in
>

// CHECK: func.func @keyswitch
func.func @keyswitch(%input: !glwe_in, %glev: !glev) -> !glwe_out {
    %glwe_switched = glwe.modulus_switch %input {
        modulus = #keyswitch_modulus,
        body = false
    } : !glwe_in -> !glwe_switched
    %glwe_decomposed = glwe.exact_decompose %glwe_switched {
        decomposition = #decomposition,
        body = false
    } : !glwe_switched -> !glwe_decomposed
    %glwe_out = glwe.exact_recompose %glwe_decomposed, %glev
      : (!glwe_decomposed, !glev) -> !glwe_out
    return %glwe_out : !glwe_out
}
