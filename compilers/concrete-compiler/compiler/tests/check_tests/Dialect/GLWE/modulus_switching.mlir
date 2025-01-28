// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

// Test full modulus_switching
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

// CHECK-LABEL: test_modulus_switching
func.func @test_modulus_switching(%arg0 : !glwe) -> !glwe_switched {
  %0 = glwe.modulus_switching %arg0 {modulus = 53} : !glwe -> !glwe_switched
  return %0 : !glwe_switched
}

!glwe_partial_switched = !glwe.glwe<
  params = #glwe.glwe_params<
    message_bound=8,
    dimension=1024,
    poly_size=2,
    mask_modulus=53,
    body_modulus=64,
    variance=0
  >
>

// CHECK-LABEL: test_modulus_switching_partial
func.func @test_modulus_switching_partial(%arg0 : !glwe) -> !glwe_partial_switched {
  %0 = glwe.modulus_switching %arg0 {modulus = 53, partial = true} : !glwe -> !glwe_partial_switched
  return %0 : !glwe_partial_switched
}