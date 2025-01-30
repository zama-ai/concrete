// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

/////////////////////////////////////////////////
// GLWE Ciphertext //////////////////////////////
/////////////////////////////////////////////////

// Define a secret key (with gaussian distribution)
#secret_key = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>
>

// Define an encoding (6 bits native)
#native_encoding = #glwe.encoding<
  body_modulus = <@m>,
  mask_modulus = <@m>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

// A unparametrized glwe ciphertext
!glwe_ciphertext = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #native_encoding
>

// CHECK: %arg0: !glwe.glwe<secret_key = <size = <dimension = <@K>, poly_size = <@N>>, distribution = <average_mean = <5.000000e-01>, average_variance = <2.500000e-01>>>, encoding = <body_modulus = <@m>, mask_modulus = <@m>, message_modulus = <2.000000e+00 ** 6.000000e+00>, right_padding = <1.000000e+00>, left_padding = <1.000000e+00>>>
func.func @test_glwe_ciphertext(%arg0: !glwe_ciphertext) {
    return
}

// Define a secret key with nb_keys != 1
#secret_key_with_nb_keys = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>,
  distribution = <average_mean=<0.5>, average_variance=<0.25>>,
  nb_keys = <3.>
>

!glwe_ciphertext_with_several_bodies = !glwe.glwe<
  secret_key = #secret_key_with_nb_keys,
  encoding = #native_encoding
>

// CHECK: %arg0: !glwe.glwe<secret_key = <size = <dimension = <@K>, poly_size = <@N>>, nb_keys = <3.000000e+00>, distribution = <average_mean = <5.000000e-01>, average_variance = <2.500000e-01>>>, encoding = <body_modulus = <@m>, mask_modulus = <@m>, message_modulus = <2.000000e+00 ** 6.000000e+00>, right_padding = <1.000000e+00>, left_padding = <1.000000e+00>>>
func.func @test_glwe_ciphertext_with_several_bodies(%arg0: !glwe_ciphertext_with_several_bodies) {
    return
}

// A unparametrized glwe ciphertext with user defined variance
!glwe_ciphertext_with_variance = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #native_encoding,
  variance = <2.0 ** (2.0 * max(@slope * (@K * @N) + @bias, 2.0 - @m))>
>

// CHECK: %arg0: !glwe.glwe<secret_key = <size = <dimension = <@K>, poly_size = <@N>>, distribution = <average_mean = <5.000000e-01>, average_variance = <2.500000e-01>>>, encoding = <body_modulus = <@m>, mask_modulus = <@m>, message_modulus = <2.000000e+00 ** 6.000000e+00>, right_padding = <1.000000e+00>, left_padding = <1.000000e+00>>>
func.func @test_glwe_ciphertext_with_variance(%arg0: !glwe_ciphertext) {
    return
}

// A resolved glwe ciphertext (example take from compiler)
!glwe_ciphertext_resolved = !glwe.glwe<
  secret_key = <
    size = <
      dimension=<1.>,
      poly_size=<512.>
    >,
    distribution = <
      average_mean=<0.5>,
      average_variance=<0.25>
    >
  >,
  encoding = <
    body_modulus = <2. ** 64.>,
    mask_modulus = <2. ** 64.>,
    message_modulus = <2. ** 6.>,
    right_padding = <1.>,
    left_padding = <1.>
  >,
  variance = <4.896863592478985e-07>
>

// CHECK: %arg0: !glwe.glwe<secret_key = <size = <dimension = <1.000000e+00>, poly_size = <5.120000e+02>>, distribution = <average_mean = <5.000000e-01>, average_variance = <2.500000e-01>>>, encoding = <body_modulus = <2.000000e+00 ** 6.400000e+01>, mask_modulus = <2.000000e+00 ** 6.400000e+01>, message_modulus = <2.000000e+00 ** 6.000000e+00>, right_padding = <1.000000e+00>, left_padding = <1.000000e+00>>, variance = <4.896863592478985E-7>>
func.func @test_glwe_ciphertext_resolved(%arg0: !glwe_ciphertext_resolved) {
    return
}