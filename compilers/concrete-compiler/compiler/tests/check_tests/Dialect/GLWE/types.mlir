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

/////////////////////////////////////////////////
// RadixGLWE Ciphertext /////////////////////////
/////////////////////////////////////////////////

// A unparametrized fully decomposed glwe ciphertext
!glwe_full_decomposed = !glwe.radix_glwe<
  params = #glwe.glwe_params<
    dimension = <@K>,
    poly_size = <@N>,
    mask_modulus = <@m>,
    body_modulus = <@m>
  >,
  decomp = #glwe.decomp<
    base_log = <@b>,
    level = <@l>
  >
>

// CHECK: %arg0: !glwe.radix_glwe<params = <dimension = <@K>, poly_size = <@N>, mask_modulus = <@m>, body_modulus = <@m>>, decomp = <base_log = <@b>, level = <@l>>>
func.func @test_radix_glwe(%arg0 : !glwe_full_decomposed) {
  return
}

// A unparametrized partially decomposed glwe ciphertext
!glwe_partial_decomposed = !glwe.radix_glwe<
  params = #glwe.glwe_params<
    dimension = <@K>,
    poly_size = <@N>,
    mask_modulus = <@m>,
    body_modulus = <@m>
  >,
  decomp = #glwe.decomp<
    base_log = <@b>,
    level = <@l>
  >,
  partial = true
>
// CHECK: %arg0: !glwe.radix_glwe<params = <dimension = <@K>, poly_size = <@N>, mask_modulus = <@m>, body_modulus = <@m>>, decomp = <base_log = <@b>, level = <@l>>, partial = true>
func.func @test_radix_glwe_partial(%arg0 : !glwe_partial_decomposed) {
  return
}

// A unparametrized fully decomposed glwe ciphertext
!glwe_full_decomposed_with_variance = !glwe.radix_glwe<
  params = #glwe.glwe_params<
    dimension = <@K>,
    poly_size = <@N>,
    mask_modulus = <@m>,
    body_modulus = <@m>
  >,
  decomp = #glwe.decomp<
    base_log = <@b>,
    level = <@l>
  >,
  variance = <2.0 ** (2.0 * max(@slope * (@K * @N) + @bias, 2.0 - @m))>
>

// CHECK: %arg0: !glwe.radix_glwe<params = <dimension = <@K>, poly_size = <@N>, mask_modulus = <@m>, body_modulus = <@m>>, decomp = <base_log = <@b>, level = <@l>>, variance = <2.000000e+00 ** (2.000000e+00 * max((@slope * (@K * @N)) + @bias, 2.000000e+00 - @m))>>
func.func @test_radix_glwe_with_variance(%arg0 : !glwe_full_decomposed_with_variance) {
  return
}

/////////////////////////////////////////////////
// RadixGLWE Ciphertext /////////////////////////
/////////////////////////////////////////////////

// A unparametrized full glev
!glev_full_decomposed = !glwe.glev<
  params = #glwe.glwe_params<
    dimension = <@K>,
    poly_size = <@N>,
    mask_modulus = <@m>,
    body_modulus = <@m>
  >,
  decomp = #glwe.decomp<
    base_log = <@b>,
    level = <@l>
  >,
  size = <@n>
>

// CHECK: %arg0: !glwe.glev<params = <dimension = <@K>, poly_size = <@N>, mask_modulus = <@m>, body_modulus = <@m>>, decomp = <base_log = <@b>, level = <@l>>, size = <@n>>
func.func @test_glev(%arg0 : !glev_full_decomposed) {
  return
}

// A unparametrized partial glev ciphertext
!glev_partial_decomposed = !glwe.glev<
  params = #glwe.glwe_params<
    dimension = <@K>,
    poly_size = <@N>,
    mask_modulus = <@m>,
    body_modulus = <@m>
  >,
  decomp = #glwe.decomp<
    base_log = <@b>,
    level = <@l>
  >,
  size = <@n>,
  partial = true
>
// CHECK: %arg0: !glwe.glev<params = <dimension = <@K>, poly_size = <@N>, mask_modulus = <@m>, body_modulus = <@m>>, decomp = <base_log = <@b>, level = <@l>>, size = <@n>, partial = true>
func.func @test_glev_partial(%arg0 : !glev_partial_decomposed) {
  return
}

// A unparametrized fully decomposed glev ciphertext
!glev_partial_decomposed_with_variance = !glwe.glev<
  params = #glwe.glwe_params<
    dimension = <@K>,
    poly_size = <@N>,
    mask_modulus = <@m>,
    body_modulus = <@m>
  >,
  decomp = #glwe.decomp<
    base_log = <@b>,
    level = <@l>
  >,
  size = <@n>,
  variance = <2.0 ** (2.0 * max(@slope * (@K * @N) + @bias, 2.0 - @m))>
>

// CHECK: %arg0: !glwe.radix_glwe<params = <dimension = <@K>, poly_size = <@N>, mask_modulus = <@m>, body_modulus = <@m>>, decomp = <base_log = <@b>, level = <@l>>, variance = <2.000000e+00 ** (2.000000e+00 * max((@slope * (@K * @N)) + @bias, 2.000000e+00 - @m))>>
func.func @glev_partial_decomposed_with_variance(%arg0 : !glwe_full_decomposed_with_variance) {
  return
}