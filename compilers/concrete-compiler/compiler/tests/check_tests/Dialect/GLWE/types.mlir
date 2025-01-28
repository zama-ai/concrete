// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

// GLWE Ciphertext //////////////////////////////

#K = #glwe.param<name="K", domain=[1, 2, 3, 4, 5, 6]>

#N = #glwe.param<name="N", domain=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]>

#gparams = #glwe.glwe_params<
  message_bound=8,
  dimension=#K,
  poly_size=#N,
  mask_modulus=64,
  body_modulus=64,
>
!glwe_ciphertext = !glwe.glwe<params = #gparams>


// CHECK-LABEL: test_glwe
func.func @test_glwe(%arg0 : !glwe_ciphertext) -> !glwe_ciphertext {
  return %arg0 : !glwe_ciphertext
}

// // Radix GLWE Ciphertext ////////////////////////
// #decomp_params = #glwe.decomp_params<
//   base_log=53,
//   level=3
// >
// !glwe_radix_ciphertext = !glwe.radix_glwe<params = #gparams, decomp=#decomp_params>

// // CHECK-LABEL: test_radix_glwe
// func.func @test_radix_glwe(%arg0 : !glwe_radix_ciphertext) -> !glwe_radix_ciphertext {
//   return %arg0 : !glwe_radix_ciphertext
// }

// !glwe_radix_ciphertext_partial = !glwe.radix_glwe<params = #gparams, decomp=#decomp_params, partial=true>
// // CHECK-LABEL: test_radix_glwe_partial
// func.func @test_radix_glwe_partial(%arg0 : !glwe_radix_ciphertext_partial) -> !glwe_radix_ciphertext_partial {
//   return %arg0 : !glwe_radix_ciphertext_partial
// }

// // GLev Ciphertext //////////////////////////////
// !glev = !glwe.glev<params = #gparams, size = 12>
// // CHECK-LABEL: test_glev
// func.func @test_glev(%arg0 : !glev) -> !glev {
//   return %arg0 : !glev
// }

// // modulus_switching ////////////////////////////

// !glwe_switched = !glwe.glwe<
//   params = #glwe.glwe_params<
//     message_bound=8,
//     dimension=2,
//     poly_size=2048,
//     mask_modulus=53,
//     body_modulus=64,
//     variance=0
//   >
// >



// // CHECK-LABEL: test_exact_decomposition
// func.func @test_exact_decomposition(%arg0 : !glwe_ciphertext) -> !glwe_radix_ciphertext {
//   %0 = glwe.exact_decomposition %arg0 {decomp = #decomp_params} : !glwe_ciphertext -> !glwe_radix_ciphertext
//   return %0 : !glwe_radix_ciphertext
// }

// // CHECK-LABEL: test_exact_recomposition
// func.func @test_exact_recomposition(%arg0 : !glwe_radix_ciphertext, %arg1 : !glev) -> !glwe_ciphertext {
//   %0 = glwe.exact_recomposition %arg0, %arg1 : (!glwe_radix_ciphertext,!glev) -> !glwe_ciphertext
//   return %0 : !glwe_ciphertext
// }

// !glwe_extracted = !glwe.glwe<
//   params = #glwe.glwe_params<
//     message_bound=8,
//     dimension=4046,
//     poly_size=1,
//     mask_modulus=64,
//     body_modulus=64,
//     variance=0
//   >
// >

// // CHECK-LABEL: sample_extract
// func.func @test_sample_extract(%arg0 : !glwe_ciphertext) -> !glwe_extracted {
//   %0 = glwe.sample_extract %arg0 : !glwe_ciphertext -> !glwe_extracted
//   return %0 : !glwe_extracted
// }