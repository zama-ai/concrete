
// # GLWE expressions 
//
// - check parser stuff (float, div, ...) (@andi)
// - check bug complex expression (@quentin)
// - implement correctly GlweBindings printer/parser (Variable) ()
// - complete the list of low level glwe operators (https://github.com/zama-ai/concrete-spec/blob/main/src/concrete/poc/low_level_operators.py)
// - first step of glwe.generic (define arbitrary cost, variance)
// - modelize glwe.constraint
//
//
// GLWE expressions are algebraic expressions that describe a crypto parameters. 
//
// Syntax
// ```
// glwe-id ::= symbol-id
// 
// glwe-expr ::=
//       `(` glwe-expr `)`
//     | glwe-expr `+` glwe-expr
//     | glwe-expr `-` glwe-expr
//     | glwe-expr `*` glwe-expr
//     | glwe-expr `**` glwe-expr
//     | glwe-expr `/` glwe-expr
//     | `-` glwe-expr
//     | `abs(` glwe-expr `)`
//     | `floor(` glwe-expr `)` 
//     | `ceil(` glwe-expr `)`
//     | `max(` glwe-expr `,` glwe-expr `)`
//     | `max(` glwe-expr `,` glwe-expr `)` // ?
//     | glwe-id
//     | float | integer
// ```
//
// and the an MLIR attribute `mlir::concretelang::GLWE::ExpressionAttr` to use in type definition and operators.
//
// ```
// #glwe.expr<glwe-expr>
// ```
//
//
// # GLWE bindings
//
// Now let's define a binding between a "defined" parameter and an expression,
// or between a "free" parameter and a domain.
//
// Syntax
// ```
// glwe-domain ::=
//      `[` float (`,` float)* `]`
//     | ... // Any syntactic sugar (level, base_log, poly_size, ...) 
// 
// glwe-let ::=
//       glwe-id `=` glwe-expr
//     | glwe-id `=` glwe-domain
// ```
//
// and the MLIR attribute `mlir::concretelang::GLWE::BindingsAttr` to binds named expression.
//
// ```
// #glwe.let<glwe-let>
// ```
//
// # GLWE constaints
//
// glwe-constraintop :== 
//  | `==` | `<` | `>` | ...
//
// ```
// #glwe.constaint<glwe-expr glwe-constraintop glwe-expr>
// ```
//
// # GLWE common parameters
//
// ## GLWEParametersAttr
//
// For factorize common parameters for GLWE types
//
// mnemonic: `glwe_params`
//
// fields:
// ```
// - dimension: ExpressionAttr
// - poly_size: ExpressionAttr
// - mask_modulus: ExpressionAttr
// - body_modulus: ExpressionAttr
// ```
//
//
// ## GLWEDecompositionAttr
//
// mnemonic: `decomp_params`
//
// fields:
// ```
// - base_log: ExpressionAttr
// - level: ExpressionAttr
// ```
//
//
// # GLWE types
//
// ## GLWE
//
// Represents a GLWE ciphertext
//
// mnemonic: `glwe`
//
// fields:
// ```
// - params: GLWEParametersAttr
// - variance: ExpressionAttr
// ```
//
//
// ## RadixGLWE
//
// Represents a GLWE ciphertext
//
// mnemonic: `radix_glwe`
//
// fields:
// ```
// - params: GLWEParametersAttr
// - decomp: GLWEDecompositionAttr
// - partial: bool
// - variance: ExpressionAttr
// ```
//
// ## GLev
//
// Represents a GLev list
//
// mnemonic: `radix_glwe`
//
// fields:
// ```
// - params: GLWEParametersAttr
// - decomp: GLWEDecompositionAttr
// - size: ExpressionAttr
// - partial: bool
// - variance: ExpressionAttr
// ```
//
// Example of binding glwe parameters and defining high level external product operator.

module attributes {
  glwe.local_perror = 0.01
  glwe.domain = #glwe.let<
    @K = [1., 2., 3., 4., 5., 6.,],
    @N = [256., 512., 1024., 2048., 4096., 8192., 16384., 32768., 65536., 131072.],
    @b = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904],
    @l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    @m = [2^32, 2^64],
    //@X = [....]
  >
} {
  !glwe_in = !glwe.glwe<
    params = #glwe.glwe_params<
      dimension = @K,
      poly_size = @N,
      mask_modulus = @m,
      body_modulus = @m,
    >
  >
  #m_switched = #glwe.expr<2 ** (b * l)>
  !glwe_switched = !glwe.glwe<
    params = #glwe.glwe_params<
      dimension = @K,
      poly_size = @N,
      mask_modulus = #m_switched,
      body_modulus = #m_switched,
    >
  >
  #decomp_params = #glwe.decomp_params<
    base_log = @b,
    level = @l
  >
  !glwe_decomposed = !glwe.radix_glwe<
    params = #glwe.glwe_params<
      dimension = @K,
      poly_size = @N,
      mask_modulus = #m_switched,
      body_modulus = #m_switched,
    >,
    decomp = #decomp_params,
    partial = false,
  >
  !glev = !glwe.glev<
    params = #glwe.glwe_params<
      dimension = @K,
      poly_size = @N,
      mask_modulus = @m,
      body_modulus = @m,
    >,
    decomp_params = #decomp_params,
    size = #glwe.expr<@K + 1>,
    partial = false
  >
  !glwe_out = !glwe.glwe<
    params = #glwe.glwe_params<
      dimension= @K,
      poly_size= @N,
      mask_modulus = @m,
      body_modulus = @m
    >
  >
  // glwe.generic 'new_op' {variance = expr, cost = expr}

  func.func @external_product(%glwe: !glwe_in, %glev: !glev) -> !glwe_out {
    %glwe_switched = glwe.modulus_switching %glwe {modulus = #m_switched, partial = false} : !glwe_in -> !glwe_switched
    ?? glwe.constraint %glwe_switched {p_error <= 0.01}
    %glwe_decomposed = glwe.exact_decomposition %switched {decomp = #decomp_params, partial = false} : !glwe_switched -> 
    !glwe_decomposed
    %glwe_out = glwe.exact_recomposition {partial = false} %decomposed, %glev : !glwe_decomposed -> !glwe_out
    return %glwe_out : !glwe_out
  }

}

python =
  def external_procitc(glwe, glvev)

  def opera(...) =
    ///
    external_

// # Variances
// 
// Each low level operations have it's own variance and cost expression,
// as example the variance for the three low level operator used above:
// 
// ## Modulus switching variance
//
// modulus_switching_variance(full) =
//   (@full + (@n * ((@q**2 / (96.0 * (@new_q / 2.0) ** 2)) + 1.0 / 48.0)))
//   / @q**2
//   + @input_variance
//
// modulus_switching_variance_partial =
//   modulus_switching_variance(0)
//
// modulus_switching_variance_full =
//   modulus_switching_variance(@q**2 / (48.0 * (@new_q / 2.0) ** 2) - 1.0 / 12.0)
//
//
// ## Exact decomposition variance
//
// exact_decomposition_variance = @input_variance
//
// ## Exact recomposition variance
//
// fft_variance =
//   @glev_level
//   * (@glev_dimension + 1)
//   * 2.0 ** (2 * @glev_base_log)
//   * 2**22
//   * 2**-2.57722494
//   * @glev_poly_size**2
//   / @glev_ciphertext_modulus**2
//   
// exact_recomposition_variance_partial =
//     @glev_size
//   * @glev_level
//   * @glev_poly_size
//   * (2 ** (@glev_base_log - 1) * (2 ** (@glev_base_log - 1) + 1))
//   / 3.0
//   * @glev_variance
//   + @input_variance + @fft_variance
//
// exact_recomposition_variance_full =
//     @glev_size
//   * @glev_level
//   * @glev_poly_size
//   * (4 * (2 ** (@glev_base_log - 1)) ** 2 - 1) / 12
//   * glev_variance
//   + @input_variance * @glev_message_bound
//
// 
// ## Minimal variance (encryption_variance)
// 
// Minimal variance is also an expression that the compiler will inject depending of the secret key and the noise distribution. This expression will contains two "free" parameter @modulus @lwe_size.
//
//
// # Compiler passes
//
// ## Variance(/cost) propagation
//
// 1. Set input variance (which is not yet set).
// Example
//
  !glwe_in = !glwe.glwe<
    params = #glwe.glwe_params<
      dimension = @K,
      poly_size = @N,
      mask_modulus = @m,
      body_modulus = @m,
    >,
    // As well minimal variance is injected by the compiler and the @modulus and @lwe_size is substitued to @m and @K * @N
    variance = minimal_variance(@m, @K * @N)
  >

  !glev = !glwe.glev<
    params = #glwe.glwe_params<
      dimension = @K,
      poly_size = @N,
      mask_modulus = @m,
      body_modulus = @m,
    >,
    decomp_params = #decomp_params,
    size = #glwe.expr<K + 1>,
    partial = false,
    variance = minimal_variance(@m, @K * @N)
  >

// 2. Propagate the variance along the dag by substitute free varaible in operator variance expressions
// Example
  !glwe_switched = !glwe.glwe<
    params = #glwe.glwe_params<
      dimension = @K,
      poly_size = @N,
      mask_modulus = @m_switched,
      body_modulus = @m_switched,
      variance = (  @m**2 / (48.0 * (@m_switched / 2.0) ** 2) - 1.0 / 12.0 
                  + ((@K * @N) * ((@m**2 / (96.0 * (@m_switched / 2.0) ** 2)) + 1.0 / 48.0)))
                  / @m**2
                  + minimal_variance(@m, @K * @N)
    >
  >

// ## Solver generation (nested loops)
// At the end of variance propagation the output will have a variance(/cost) expression with free parameter, that we can transform into mlir standard function.
// Example
func.func @external_product_variance(%K: f64, %N: f64, %b: f64, %l: f64) -> f64 {
  ...
}

func.func @external_product_cost(%K: f64, %N: f64, %b: f64, %l: f64) -> f64 {
  ...
}

// and for all free parameter in noise/cost expression we can generate the nested loops solver
func solve_external_product(%p_error: f64) -> f64, f64, f64, f64 {
  %K_domain = arith.constant dense<[1., 2., 3., 4., 5., 6.]> : tensor<6xf64>
  %N_domain = [256., 512., 1024., 2048., 4096., 8192., 16384., 32768., 65536., 131072.],
  %b_domain = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904],
  %l_domain = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
  %m_domain = [2^32, 2^64]
  %variance_bound = @variance_bound(%p_error, ...) : f64
  for %i = 0 to 6 step 1 {
    %K = tensor.extract %K_domain, %i : tensor<6xf64>
    for ... {
      ...
      ...

      %variance = @external_product_variance(%K, %N, %b, %l) : f64
      %is_satisfied = arith.cmpf olt, %variance, %variance_bound : f64
      ...
    }
  }
}

// ## Solver generation any other...

// ## Code generation
//
// As well from the specification we could trivially generate an actual code, mlir, rust ... and from the solver we could generate the parameters etc...
//
