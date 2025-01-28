// RUN: concretecompiler --split-input-file --verify-diagnostics %s 

!glwe = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1024,
    poly_size=2,
    mask_modulus=64,
    body_modulus=64,
    variance=0
  >
>

!glwe_switched  = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1023,
    poly_size=2,
    mask_modulus=53,
    body_modulus=53,
    variance=0
  >
>

func.func @test_modulus_switching_input_params_not_match_ouput(%arg0 : !glwe) -> !glwe_switched {
  // expected-error @+1 {{'glwe.modulus_switching' op failed to verify that the input GLWE {message_bound, dimension, poly_size} parameters are equals to the output value}}
  %0 = glwe.modulus_switching %arg0 {modulus = 53} : !glwe -> !glwe_switched
  return %0 : !glwe_switched
}

// -----

// Bad ouput mask_modulus

!glwe = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1024,
    poly_size=2,
    mask_modulus=64,
    body_modulus=64,
    variance=0
  >
>

!glwe_switched  = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1024,
    poly_size=2,
    mask_modulus=52,
    body_modulus=53,
    variance=0
  >
>

func.func @test_modulus_switching_mask_modulus(%arg0 : !glwe) -> !glwe_switched {
  // expected-error @+1 {{'glwe.modulus_switching' op failed to verify that {modulus} parameter is equals to the output GLWE {mask_modulus} parameter}}
  %0 = glwe.modulus_switching %arg0 {modulus = 53} : !glwe -> !glwe_switched
  return %0 : !glwe_switched
}

// -----

// Bad ouput body_modulus

!glwe = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1024,
    poly_size=2,
    mask_modulus=64,
    body_modulus=64,
    variance=0
  >
>

!glwe_switched  = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1024,
    poly_size=2,
    mask_modulus=53,
    body_modulus=52,
    variance=0
  >
>

func.func @test_modulus_switching_body_modulus(%arg0 : !glwe) -> !glwe_switched {
  // expected-error @+1 {{'glwe.modulus_switching' op failed to verify that {modulus} parameter is equals to the output GLWE {body_modulus} parameter}}
  %0 = glwe.modulus_switching %arg0 {modulus = 53} : !glwe -> !glwe_switched
  return %0 : !glwe_switched
}

// -----

// Bad ouput body_modulus for partial modulus_sitching

!glwe = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1024,
    poly_size=2,
    mask_modulus=64,
    body_modulus=64,
    variance=0
  >
>

!glwe_switched  = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1024,
    poly_size=2,
    mask_modulus=53,
    body_modulus=53,
    variance=0
  >
>

func.func @test_modulus_switching_body_modulus(%arg0 : !glwe) -> !glwe_switched {
  // expected-error @+1 {{'glwe.modulus_switching' op with {partial = true} failed to verify that the input GLWE {body_modulus} parameter is equal to the output value}}
  %0 = glwe.modulus_switching %arg0 {modulus = 53, partial = true} : !glwe -> !glwe_switched
  return %0 : !glwe_switched
}