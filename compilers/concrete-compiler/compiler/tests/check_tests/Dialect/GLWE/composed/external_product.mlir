// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s

// Test a specification of an external product

// Define the decomposition with free variables @b and @l,
// respectivly for base_log and level parameters of the decomposition 
#decomposition = #glwe.decomposition<
  base_log = <@b>,
  level = <@l>
>

// Define the external product modulus as @b power of @l
#external_product_modulus = #glwe.expr<@b ** @l>

// Define the input secret key with a fixed distribution (gaussian)
// and free variables @K and @N respectively for the glwe dimesion and the size of the polynomial.
#secret_key = #glwe.secret_key<
  size = <dimension=<@K>, poly_size=<@N>>,
  distribution = <Binary>
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
  secret_key = #secret_key,
  encoding = #encoding_in
>

// Define the encoding of the GLWE switched ciphertex 
#encoding_switched = #glwe.encoding<
  body_modulus = #external_product_modulus,
  mask_modulus = #external_product_modulus,
  message_modulus = <2. ** 6.>,
  right_padding = <1.>,
  left_padding = <1.>
>

// Define the type of the GLWE switched ciphertextt
!glwe_switched = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #encoding_switched
>

// Define the type of the GLWE decomposed ciphertext
!glwe_decomposed = !glwe.radix_glwe<
  secret_key = #secret_key,
  encoding = #encoding_switched,
  decomposition = #decomposition
>

// Define the type of glev for the recomposition
!glev = !glwe.glev<
  secret_key = #secret_key,
  encoding = #encoding_in,
  decomposition = #decomposition,
  shape = (<@K + 1.>),
  average_message = <0.5>
>

!glwe_out = !glwe.glwe<
  secret_key = #secret_key,
  encoding = #encoding_in
>

module attributes {
  // Example of unsolved domain
  //glwe.N = #glwe.domain<@N in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]>,
  //glwe.K = #glwe.domain<@K in [1, 2, 3, 4, 5, 6]>,
  //glwe.l = #glwe.domain<@l in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]>,
  //glwe.b = #glwe.domain<@b in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904]>
  // Example of solved domain (after optimization)
  // Golden ref => https://github.com/zama-ai/concrete-spec/blob/075ed6dd8920dd3fabb22c83f43af4062eb9c53a/tests/test_keyswitch.py#L27
  glwe.N = #glwe.domain<@N in [512]>,
  glwe.K = #glwe.domain<@K in [4]>,
  glwe.l = #glwe.domain<@l in [22]>,
  glwe.b = #glwe.domain<@b in [4]>
} {
// CHECK: func.func @external_product
func.func @external_product(%input: !glwe_in, %glev: !glev) -> !glwe_out {
    %glwe_switched = glwe.modulus_switch %input {
        modulus = #external_product_modulus
    } : !glwe_in -> !glwe_switched
    %glwe_decomposed = glwe.exact_decompose %glwe_switched {
        decomposition = #decomposition
    } : !glwe_switched -> !glwe_decomposed
    %glwe_out = glwe.exact_recompose %glwe_decomposed, %glev {fft = true}
      : (!glwe_decomposed, !glev) -> !glwe_out
    // CHECK: return {{.*}} : !glwe.glwe<{{.*}}variance = <1.1447461{{[0-9]+}}E-24>>
    return %glwe_out : !glwe_out
}
}