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
  variance = <@input_variance>
>

module attributes {
} {
// CHECK: func.func @dot
func.func @dot(%glwes: tensor<3x!glwe>) -> !glwe {
    %glwe_out = glwe.dot %glwes {
        weights = [1, #glwe.expr<@W>, 2.]
    } : tensor<3x!glwe> -> !glwe
    // CHECK: return {{.*}} : !glwe.glwe<{{.*}}variance = <((1.000000e+00 + @W) + 2.000000e+00) * @input_variance>>
    return %glwe_out : !glwe
}
}
