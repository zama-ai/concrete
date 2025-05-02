<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.configuration`
Declaration of `Configuration` class. 

**Global Variables**
---------------
- **MAXIMUM_TLU_BIT_WIDTH**
- **DEFAULT_P_ERROR**
- **DEFAULT_GLOBAL_P_ERROR**


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SecurityLevel`
Security level used to optimize the circuit parameters. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ParameterSelectionStrategy`
ParameterSelectionStrategy, to set optimization strategy. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiParameterStrategy`
MultiParamStrategy, to set optimization strategy for multi-parameter. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Exactness`
Exactness, to specify for specific operator the implementation preference (default and local). 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ApproximateRoundingConfig`
Controls the behavior of approximate rounding. 

In the following `k` is the ideal rounding output precision. Often the precision used after rounding is `k`+1 to avoid overflow. `logical_clipping`, `approximate_clipping_start_precision` can be used to stay at precision `k`, either logically or physically at the successor TLU. See examples in https://github.com/zama-ai/concrete/blob/main/docs/core-features/rounding.md. 

<a href="../../<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    logical_clipping: bool = True,
    approximate_clipping_start_precision: int = 5,
    reduce_precision_after_approximate_clipping: bool = True,
    symetrize_deltas: bool = True
) → None
```









---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ComparisonStrategy`
ComparisonStrategy, to specify implementation preference for comparisons. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BitwiseStrategy`
BitwiseStrategy, to specify implementation preference for bitwise operations. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L710"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultivariateStrategy`
MultivariateStrategy, to specify implementation preference for multivariate operations. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L819"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MinMaxStrategy`
MinMaxStrategy, to specify implementation preference for minimum and maximum operations. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L954"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Configuration`
Configuration class, to allow the compilation process to be customized. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L1018"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    verbose: bool = False,
    show_graph: Optional[bool] = None,
    show_bit_width_constraints: Optional[bool] = None,
    show_bit_width_assignments: Optional[bool] = None,
    show_assigned_graph: Optional[bool] = None,
    show_mlir: Optional[bool] = None,
    show_optimizer: Optional[bool] = None,
    show_statistics: Optional[bool] = None,
    dump_artifacts_on_unexpected_failures: bool = True,
    enable_unsafe_features: bool = False,
    use_insecure_key_cache: bool = False,
    insecure_key_cache_location: Optional[Path, str] = None,
    loop_parallelize: bool = True,
    dataflow_parallelize: bool = False,
    auto_parallelize: bool = False,
    compress_evaluation_keys: bool = False,
    compress_input_ciphertexts: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    auto_adjust_rounders: bool = False,
    auto_adjust_truncators: bool = False,
    single_precision: bool = False,
    parameter_selection_strategy: Union[ParameterSelectionStrategy, str] = <ParameterSelectionStrategy.MULTI: 'multi'>,
    multi_parameter_strategy: Union[MultiParameterStrategy, str] = <MultiParameterStrategy.PRECISION: 'precision'>,
    show_progress: bool = False,
    progress_title: str = '',
    progress_tag: Union[bool, int] = False,
    fhe_simulation: bool = False,
    fhe_execution: bool = True,
    compiler_debug_mode: bool = False,
    compiler_verbose_mode: bool = False,
    comparison_strategy_preference: Optional[ComparisonStrategy, str, list[Union[ComparisonStrategy, str]]] = None,
    bitwise_strategy_preference: Optional[BitwiseStrategy, str, list[Union[BitwiseStrategy, str]]] = None,
    shifts_with_promotion: bool = True,
    multivariate_strategy_preference: Optional[MultivariateStrategy, str, list[Union[MultivariateStrategy, str]]] = None,
    min_max_strategy_preference: Optional[MinMaxStrategy, str, list[Union[MinMaxStrategy, str]]] = None,
    composable: bool = False,
    use_gpu: bool = False,
    relu_on_bits_threshold: int = 7,
    relu_on_bits_chunk_size: int = 3,
    if_then_else_chunk_size: int = 3,
    additional_pre_processors: Optional[list[GraphProcessor]] = None,
    additional_post_processors: Optional[list[GraphProcessor]] = None,
    rounding_exactness: Exactness = <Exactness.EXACT: 'exact'>,
    approximate_rounding_config: Optional[ApproximateRoundingConfig] = None,
    optimize_tlu_based_on_measured_bounds: bool = False,
    enable_tlu_fusing: bool = True,
    print_tlu_fusing: bool = False,
    optimize_tlu_based_on_original_bit_width: Union[bool, int] = 8,
    detect_overflow_in_simulation: bool = False,
    dynamic_indexing_check_out_of_bounds: bool = True,
    dynamic_assignment_check_out_of_bounds: bool = True,
    simulate_encrypt_run_decrypt: bool = False,
    range_restriction: Optional[RangeRestriction] = None,
    keyset_restriction: Optional[KeysetRestriction] = None,
    auto_schedule_run: bool = False,
    security_level: SecurityLevel = <SecurityLevel.SECURITY_128_BITS: 128>,
    optim_lsbs_with_lut: bool = True
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/configuration.py#L1213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fork`

```python
fork(
    verbose: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    show_graph: Optional[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    show_bit_width_constraints: Optional[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    show_bit_width_assignments: Optional[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    show_assigned_graph: Optional[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    show_mlir: Optional[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    show_optimizer: Optional[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    show_statistics: Optional[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    dump_artifacts_on_unexpected_failures: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    enable_unsafe_features: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    use_insecure_key_cache: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    insecure_key_cache_location: Optional[Keep, Path, str] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    loop_parallelize: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    dataflow_parallelize: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    auto_parallelize: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    compress_evaluation_keys: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    compress_input_ciphertexts: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    p_error: Optional[Keep, float] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    global_p_error: Optional[Keep, float] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    auto_adjust_rounders: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    auto_adjust_truncators: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    single_precision: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    parameter_selection_strategy: Union[Keep, ParameterSelectionStrategy, str] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    multi_parameter_strategy: Union[Keep, MultiParameterStrategy, str] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    show_progress: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    progress_title: Union[Keep, str] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    progress_tag: Union[Keep, bool, int] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    fhe_simulation: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    fhe_execution: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    compiler_debug_mode: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    compiler_verbose_mode: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    comparison_strategy_preference: Optional[Keep, ComparisonStrategy, str, list[Union[ComparisonStrategy, str]]] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    bitwise_strategy_preference: Optional[Keep, BitwiseStrategy, str, list[Union[BitwiseStrategy, str]]] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    shifts_with_promotion: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    multivariate_strategy_preference: Optional[Keep, MultivariateStrategy, str, list[Union[MultivariateStrategy, str]]] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    min_max_strategy_preference: Optional[Keep, MinMaxStrategy, str, list[Union[MinMaxStrategy, str]]] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    composable: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    use_gpu: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    relu_on_bits_threshold: Union[Keep, int] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    relu_on_bits_chunk_size: Union[Keep, int] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    if_then_else_chunk_size: Union[Keep, int] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    additional_pre_processors: Optional[Keep, list[GraphProcessor]] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    additional_post_processors: Optional[Keep, list[GraphProcessor]] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    rounding_exactness: Union[Keep, Exactness] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    approximate_rounding_config: Optional[Keep, ApproximateRoundingConfig] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    optimize_tlu_based_on_measured_bounds: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    enable_tlu_fusing: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    print_tlu_fusing: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    optimize_tlu_based_on_original_bit_width: Union[Keep, bool, int] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    detect_overflow_in_simulation: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    dynamic_indexing_check_out_of_bounds: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    dynamic_assignment_check_out_of_bounds: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    simulate_encrypt_run_decrypt: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    range_restriction: Optional[Keep, RangeRestriction] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    keyset_restriction: Optional[Keep, KeysetRestriction] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    auto_schedule_run: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    security_level: Union[Keep, SecurityLevel] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>,
    optim_lsbs_with_lut: Union[Keep, bool] = <concrete.fhe.compilation.configuration.Configuration.Keep object at ADDRESS>
) → Configuration
```

Get a new configuration from another one specified changes. 

See Configuration. 


