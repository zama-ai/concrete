# Configure

This document provides instructions on how to customize the compilation pipeline using `Configuration`s in Python and describes various configuration options available.

You can customize **Concrete** using the `fhe.Configuration` :

```python
from concrete import fhe
import numpy as np

configuration = fhe.Configuration(p_error=0.01, dataflow_parallelize=True)

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x + 42

inputset = range(10)
circuit = f.compile(inputset, configuration=configuration)
```

You can overwrite individual configuration options by specifying kwargs in the `compile` method:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x + 42

inputset = range(10)
circuit = f.compile(inputset, p_error=0.01, dataflow_parallelize=True)
```

You can also combine both ways:

```python
from concrete import fhe
import numpy as np

configuration = fhe.Configuration(p_error=0.01)

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x + 42

inputset = range(10)
circuit = f.compile(inputset, configuration=configuration, loop_parallelize=True)
```

{% hint style="info" %}

When options are specified both in the `configuration` and as kwargs in the `compile` method, the kwargs take precedence. 

{% endhint %}

## Options

#### approximate_rounding_config: ApproximateRoundingConfig = fhe.ApproximateRoundingConfig()
- Provide fine control for [approximate rounding](../core-features/rounding.md#approximate-rounding-features):
  - To enable exact clipping,
  - Or/and approximate clipping, which makes overflow protection faster.

#### auto_adjust_rounders: bool = False
- Adjust rounders automatically.

#### auto_parallelize: bool = False
- Enable auto parallelization in the compiler.

#### bitwise_strategy_preference: Optional[Union[BitwiseStrategy, str, List[Union[BitwiseStrategy, str]]]] = None
- Specify preference for bitwise strategies, can be a single strategy or an ordered list of strategies. See [Bitwise](../core-features/bitwise.md) to learn more.

#### compiler_debug_mode: bool = False
- Enable or disable the debug mode of the compiler. This can show a lot of information, including passes and pattern rewrites.

#### compiler_verbose_mode: bool = False
- Enable or disable verbose mode of the compiler. This mainly shows logs from the compiler and is less verbose than the debug mode.

#### comparison_strategy_preference: Optional[Union[ComparisonStrategy, str, List[Union[ComparisonStrategy, str]]]] = None
- Specify preference for comparison strategies. Can be a single strategy or an ordered list of strategies. See [Comparisons](../core-features/comparisons.md) to learn more.

#### compress_evaluation_keys: bool = False
- Specify that serialization takes the compressed form of evaluation keys.

#### compress_input_ciphertexts: bool = False
- Specify that serialization takes the compressed form of input ciphertexts.

#### composable: bool = False
- Specify that the function must be composable with itself. 
- Only used when compiling a single circuit; when compiling modules, use the [composition policy](../compilation/composing_functions_with_modules.md#optimizing_runtimes_with_composition_policies).

#### dataflow_parallelize: bool = False
- Enable dataflow parallelization in the compiler.

#### dump_artifacts_on_unexpected_failures: bool = True
- Export debugging artifacts automatically on compilation failures.

#### enable_tlu_fusing: bool = True
- Enables Table Lookups(TLU) fusing to reduce the number of TLUs.

#### enable_unsafe_features: bool = False
- Enable unsafe features.

#### fhe_execution: bool = True
- Enable FHE execution. Can be enabled later using `circuit.enable_fhe_execution()`.

#### fhe_simulation: bool = False
- Enable FHE simulation. Can be enabled later using `circuit.enable_fhe_simulation()`.

#### global_p_error: Optional[float] = None
- Global error probability for the whole circuit. 
- If set, the whole circuit will have the probability of a non-exact result smaller than the set value. See [Exactness](../core-features/table_lookups_advanced.md#table-lookup-exactness) to learn more.

#### if_then_else_chunk_size: int = 3
- Chunk size to use when converting the `fhe.if_then_else extension`.

#### insecure_key_cache_location: Optional[Union[Path, str]] = None
- Location of insecure key cache.

#### loop_parallelize: bool = True
- Enable loop parallelization in the compiler.

#### multi_parameter_strategy: fhe.MultiParameterStrategy = fhe.MultiParameterStrategy.PRECISION
- Set the level of circuit partitioning when using `fhe.ParameterSelectionStrategy.MULTI`.
  - `PRECISION`: all TLUs with the same input precision have their own parameters.
  - `PRECISION_AND_NORM2`: all TLUs with the same input precision and output [norm2](../../compilers/concrete-optimizer/v0-parameters/) have their own parameters.

#### optimize_tlu_based_on_measured_bounds: bool = False
- Enables TLU optimizations based on measured bounds. 
- Not enabled by default, as it could result in unexpected overflows during runtime.

#### optimize_tlu_based_on_original_bit_width: Union[bool, int] = 8
- Configures whether to convert values to their original precision before doing a table lookup on them.
  - `True` enables it for all cases.
  - `False` disables it for all cases.
  - Integer value enables or disables it depending on the original bit width. With the default value of 8, only the values with original bit width ≤ 8 will be converted to their original precision.

#### p_error: Optional[float] = None
- Error probability for individual table lookups.
- If set, all table lookups will have the probability of a non-exact result smaller than the set value. See [Exactness](../core-features/table_lookups_advanced.md#table-lookup-exactness) to learn more.

#### parameter_selection_strategy: fhe.ParameterSelectionStrategy = fhe.ParameterSelectionStrategy.MULTI
- Set how cryptographic parameters are selected.

#### print_tlu_fusing: bool = False
- Enables printing of TLU fusing to see which table lookups are fused.

#### progress_tag: Union[bool, int] = False
- How many nested tag elements to display with the progress bar. 
  - `True` means all tag elements
  - `False` disables the display. 
  - `2` will display `elmt1.elmt2`.
#### progress_title: str = ""
- Title of the progress bar.

#### rounding_exactness: Exactness = fhe.Exactness.EXACT
- Set default exactness mode for the rounding operation:
  - `EXACT`: threshold for rounding up or down is exactly centered between the upper and lower value.
  - `APPROXIMATE`: faster but threshold for rounding up or down is approximately centered with a pseudo-random shift. Precise behavior is described in [`fhe.rounding_bit_pattern`](../core-features/rounding.md).

#### relu_on_bits_chunk_size: int = 3
- Chunk size of the ReLU extension when [fhe.bits](../core-features/bit_extraction.md) implementation is used.

#### relu_on_bits_threshold: int = 7
- Bit-width to start implementing the ReLU extension with [fhe.bits](../core-features/bit_extraction.md).

#### shifts_with_promotion: bool = True
- Enable promotions in encrypted shifts instead of casting at runtime. See [Bitwise#Shifts](../core-features/bitwise.md#shifts) to learn more.

#### show_graph: Optional[bool] = None
- Print computation graph during compilation. 
  - `True` means always print
  - `False` means never print
  - `None` means print depending on verbose configuration.

#### show_mlir: Optional[bool] = None
- Print MLIR during compilation. 
  - `True` means always print
  - `False` means never print
  - `None` means print depending on verbose configuration.
#### show_optimizer: Optional[bool] = None
- Print optimizer output during compilation. 
  - `True` means always print
  - `False` means never print
  - `None` means print depending on verbose configuration.
#### show_progress: bool = False
- Display a progress bar during circuit execution.

#### show_statistics: Optional[bool] = None
- Print circuit statistics during compilation. 
  - `True` means always print
  - `False` means never print
  - `None` means print depending on verbose configuration.

#### simulate_encrypt_run_decrypt: bool = False
- Whether to use the simulate encrypt/run/decrypt methods of the circuit/module instead of actual encryption/evaluation/decryption.
  - When this option is set to `True`, encrypt and decrypt are identity functions, and run is a wrapper around simulation. In other words, this option allows switching off encryption to quickly test if a function has the expected semantic (without paying the price of FHE execution).
  - This is extremely unsafe and should only be used during development.
  - For this reason, it requires `enable_unsafe_features` to be set to `True`.

#### single_precision: bool = False
- Use single precision for the whole circuit.

#### range_restriction: Optional[RangeRestriction] = None
- A range restriction to pass to the optimizer to restrict the available crypto-parameters.

#### keyset_restriction: Optional[KeysetRestriction] = None
- A keyset restriction to pass to the optimizer to restrict the available crypto-parameters.

#### use_gpu: bool = False
- Enable generating code for GPU in the compiler.

#### use_insecure_key_cache: bool = False (Unsafe)
- Use the insecure key cache.

#### verbose: bool = False
- Print details related to compilation.

#### auto_schedule_run: bool = False
  - Enable automatic scheduling of `run` method calls. When enabled, fhe function are computated in parallel in a background threads pool. When several `run` are composed, they are automatically synchronized.
  - For now, it only works for the `run` method of a `FheModule`, in that case you obtain a `Future[Value]` immediately instead of a `Value` when computation is finished.
  - E.g. `my_module.f3.run( my_module.f1.run(a), my_module.f1.run(b) )` will runs `f1` and `f2` in parallel in the background and `f3` in background when both `f1` and `f2` intermediate results are available.
  - If you want to manually synchronize on the termination of a full computation, e.g. you want to return the encrypted result, you can call explicitely `value.result()` to wait for the result. To simplify testing, decryption does it automatically.
  - Automatic scheduling behavior can be override locally by calling directly a variant of `run`:
    - `run_sync`: forces the fhe function to occur in the current thread, not in the background,
    - `run_async`: forces the fhe function to occur in a background thread, returning immediately a `Future[Value]`

#### security_level: int = 128
- Set the level of security used to perform the optimization of crypto-parameters.
