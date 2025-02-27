// RUN: concretecompiler --action=glwe-optimize %s  2>&1| FileCheck %s

// Test the generic operator

// Define a secret key
#secret_key = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>,
  distribution = <Binary>
>

//Define an encoding
#encoding = #glwe.encoding<
  body_modulus = < 2. ** 64.>,
  mask_modulus = < 2. ** 64.>,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

// Define the GLWE input ciphertext type
!glwe = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #encoding,
  variance = <@glwe_variance>
>

// Define the type of the GLWE decomposed ciphertext
!radix_glwe = !glwe.radix_glwe<
  secret_key = #secret_key,
  encoding = #encoding,
  decomposition = <base = <@br>, level = <@lr>>,
  variance = <@radix_glwe_variance>
>

// Define the type of glev for the recomposition
!glev = !glwe.glev<
  secret_key = #secret_key,
  encoding = #encoding,
  decomposition = <base = <@b>, level = <@l>>,
  shape = (<@K>),
  average_message = <1.>,
  variance = <@glev_variance>
>

module attributes {
  glwe.N = #glwe.domain<@N in [16384]>,
  glwe.glwe_variance = #glwe.domain<@glwe_variance in [1.2]>,
  glwe.radix_glwe_variance = #glwe.domain<@radix_glwe_variance in [2.8]>,
  glwe.glev_variance = #glwe.domain<@glev_variance in [4.]>
} {
// CHECK: func.func @generic
func.func @generic(%glwe: !glwe, %radix: !radix_glwe, %glev: !glev) -> !glwe {
    // Let define a generic glwe operator with a variance formula, taking expression from inputs
    %glwe_out = glwe.generic %glwe, %radix, %glev {
        variance = #glwe.expr<@self.in0.secret_key.size.poly_size * (@self.in0.variance + @self.in1.variance + @self.in2.variance)>
    } : !glwe, !radix_glwe, !glev -> !glwe
    // CHECK: return {{.*}} : !glwe.glwe<{{.*}}variance = <1.310720e+05>>
    return %glwe_out : !glwe
}
}
