// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

/////////////////////////////////////////////////
// GLWE Ciphertext //////////////////////////////
/////////////////////////////////////////////////

// A unparametrized glwe ciphertext
#gparams = #glwe.glwe_params<
  dimension = <@K>,
  poly_size = <@N>,
  mask_modulus = <@m>,
  body_modulus = <@m>
>
!glwe_ciphertext = !glwe.glwe<params = #gparams>

// CHECK: %arg0: !glwe.glwe<params = <dimension = <@K>, poly_size = <@N>, mask_modulus = <@m>, body_modulus = <@m>>>
func.func @test_glwe_ciphertext_00(%arg0: !glwe_ciphertext) {
    return
}

// A unparametrized glwe ciphertext with minimal variance
!glwe_ciphertext_with_minimal_variance = !glwe.glwe<
    params = #gparams,
    variance = <2.0 ** (2.0 * max(@slope * (@K * @N) + @bias, 2.0 - @m))>
>

// CHECK: %arg0: !glwe.glwe<params = <dimension = <@K>, poly_size = <@N>, mask_modulus = <@m>, body_modulus = <@m>>, variance = <2.000000e+00 ** (2.000000e+00 * max((@slope * (@K * @N)) + @bias, 2.000000e+00 - @m))>>
func.func @test_glwe_ciphertext_with_minimal_variance(%arg0: !glwe_ciphertext_with_minimal_variance) {
    return
}

// A resolved glwe ciphertext (parameters example taken from compiler)
!glwe_ciphertext_resolved = !glwe.glwe<
  params = #glwe.glwe_params<
    dimension = <1.>,
    poly_size = <512.>,
    mask_modulus = <64.>,
    body_modulus = <64.>
  >,
  variance = <4.896863592478985e-07>
>
// CHECK: %arg0: !glwe.glwe<params = <dimension = <1.000000e+00>, poly_size = <5.120000e+02>, mask_modulus = <6.400000e+01>, body_modulus = <6.400000e+01>>, variance = <4.896863592478985E-7>>
func.func @test_glwe_ciphertext_resolved(%arg0: !glwe_ciphertext_resolved) {
    return
}
