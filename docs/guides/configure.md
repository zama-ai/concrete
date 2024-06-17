# Configure

**Concrete** can be customized using `Configuration`s:

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

You can overwrite individual options as kwargs to the `compile` method:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x + 42

inputset = range(10)
circuit = f.compile(inputset, p_error=0.01, dataflow_parallelize=True)
```

Or you can combine both:

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
Additional kwargs to `compile` functions take higher precedence. So if you set the option in both `configuration` and `compile` methods, the value in the `compile` method will be used.
{% endhint %}

## Options

* **show\_graph**: Optional\[bool] = None
  * Print computation graph during compilation. `True` means always print, `False` means never print, `None` means print depending on verbose configuration below.
* **show\_mlir**: Optional\[bool] = None
  * Print MLIR during compilation. `True` means always print, `False` means never print, `None` means print depending on verbose configuration below.
* **show\_optimizer**: Optional\[bool] = None
  * Print optimizer output during compilation. `True` means always print, `False` means never print, `None` means print depending on verbose configuration below.
* **show\_statistics**: Optional\[bool] = None
  * Print circuit statistics during compilation. `True` means always print, `False` means never print, `None` means print depending on verbose configuration below.
* **verbose**: bool = False
  * Print details related to compilation.
* **dump\_artifacts\_on\_unexpected\_failures**: bool = True
  * Export debugging artifacts automatically on compilation failures.
* **auto\_adjust\_rounders**: bool = False
  * Adjust rounders automatically.
* **p\_error**: Optional\[float] = None
  * Error probability for individual table lookups. If set, all table lookups will have the probability of a non-exact result smaller than the set value. See [Exactness](../core-features/table\_lookups.md#table-lookup-exactness) to learn more.
* **global\_p\_error**: Optional\[float] = None
  * Global error probability for the whole circuit. If set, the whole circuit will have the probability of a non-exact result smaller than the set value. See [Exactness](../core-features/table\_lookups.md#table-lookup-exactness) to learn more.
* **single\_precision**: bool = False
  * Use single precision for the whole circuit.
* **parameter\_selection\_strategy**: (fhe.ParameterSelectionStrategy) = fhe.ParameterSelectionStrategy.MULTI
  * Set how cryptographic parameters are selected.
* **multi\_parameter\_strategy**: fhe.MultiParameterStrategy = fhe.MultiParameterStrategy.PRECISION
  * Set the level of circuit partionning when using `fhe.ParameterSelectionStrategy.MULTI`.
  * `PRECISION`: all TLU with same input precision have their own parameters.
  * `PRECISION_AND_NORM2`: all TLU with same input precision and output [norm2](../../compilers/concrete-optimizer/v0-parameters/) have their own parameters.
* **loop\_parallelize**: bool = True
  * Enable loop parallelization in the compiler.
* **dataflow\_parallelize**: bool = False
  * Enable dataflow parallelization in the compiler.
* **auto\_parallelize**: bool = False
  * Enable auto parallelization in the compiler.
* **use_gpu**: bool = False
  * Enable generating code for GPU in the compiler.
* **enable\_unsafe\_features**: bool = False
  * Enable unsafe features.
* **use\_insecure\_key\_cache**: bool = False _(Unsafe)_
  * Use the insecure key cache.
* **insecure\_key\_cache\_location**: Optional\[Union\[Path, str]] = None
  * Location of insecure key cache.
* **show\_progress**: bool = False,
  * Display a progress bar during circuit execution
* **progress\_title**: str = "",
  * Title of the progress bar
* **progress\_tag**: Union\[bool, int] = False,
  * How many nested tag elements to display with the progress bar. `True` means all tag elements and `False` disables the display. `2` will display `elmt1.elmt2`
* **fhe\_simulation**: bool = False
  * Enable FHE simulation. Can be enabled later using `circuit.enable_fhe_simulation()`.
* **fhe\_execution**: bool = True
  * Enable FHE execution. Can be enabled later using `circuit.enable_fhe_execution()`.
* **compiler\_debug\_mode**: bool = False,
  * Enable/disable debug mode of the compiler. This can show a lot of information, including passes and pattern rewrites.
* **compiler\_verbose\_mode**: bool = False,
  * Enable/disable verbose mode of the compiler. This mainly shows logs from the compiler, and is less verbose than the debug mode.
* **comparison\_strategy\_preference**: Optional\[Union\[ComparisonStrategy, str, List\[Union\[ComparisonStrategy, str]]]] = None
  * Specify preference for comparison strategies, can be a single strategy or an ordered list of strategies. See [Comparisons](../core-features/comparisons.md) to learn more.
* **bitwise\_strategy\_preference**: Optional\[Union\[BitwiseStrategy, str, List\[Union\[BitwiseStrategy, str]]]] = None
  * Specify preference for bitwise strategies, can be a single strategy or an ordered list of strategies. See [Bitwise](../core-features/bitwise.md) to learn more.
* **shifts\_with\_promotion**: bool = True,
  * Enable promotions in encrypted shifts instead of casting in runtime. See [Bitwise#Shifts](../core-features/bitwise.md#Shifts) to learn more.
* **composable**: bool = False,
  * Specify that the function must be composable with itself. Only used when compiling a single circuit; when compiling modules use the [composition policy](../compilation/composing_functions_with_modules.md#optimizing_runtimes_with_composition_policies).
* **relu\_on\_bits\_threshold**: int = 7,
  * Bit-width to start implementing the ReLU extension with [fhe.bits](../core-features/bit\_extraction.md).
* **relu\_on\_bits\_chunk\_size**: int = 3,
  * Chunk size of the ReLU extension when [fhe.bits](../core-features/bit\_extraction.md) implementation is used.
* **if\_then\_else\_chunk\_size**: int = 3
  * Chunk size to use when converting `fhe.if_then_else` extension.
* **rounding\_exactness** : Exactness = `fhe.Exactness.EXACT`
  * Set default exactness mode for the rounding operation:
  * `EXACT`: threshold for rounding up or down is exactly centered between upper and lower value,
  * `APPROXIMATE`: faster but threshold for rounding up or down is approximately centered with pseudo-random shift.
  * Precise and more complete behavior is described in [fhe.rounding\_bit\_pattern](../core-features/rounding.md).
* **approximate\_rounding\_config** : ApproximateRoundingConfig = `fhe.ApproximateRoundingConfig()`:
  * Provide more fine control on [approximate rounding](../core-features/rounding.md#approximate-rounding-features):
  * to enable exact cliping,
  * or/and approximate clipping which make overflow protection faster.
* **optimize_tlu_based_on_measured_bounds** : bool = False
  * Enables TLU optimizations based on measured bounds.
  * Not enabled by default as it could result in unexpected overflows during runtime.
* **enable_tlu_fusing** : bool = True
  * Enables TLU fusing to reduce the number of table lookups.
* **print_tlu_fusing** : bool = False
  * Enables printing TLU fusing to see which table lookups are fused.
* **compress\_evaluation\_keys**: bool = False,
  * This specifies that serialization takes the compressed form of evaluation keys.
* **compress\_input\_ciphertexts**: bool = False,
  * This specifies that serialization takes the compressed form of input ciphertexts.
* **optimize\_tlu\_based\_on\_original\_bit\_width**: Union\[bool, int] = 8,
  * Configures whether to convert values to their original precision before doing a table lookup on them.
  * True enables it for all cases.
  * False disables it for all cases.
  * Integer value enables or disables it depending on the original bit width.
  * With the default value of 8, only the values with original bit width <= 8 will be converted to their original precision.
* **simulate\_encrypt\_run\_decrypt**: bool = False
  * Whether to use simulate encrypt/run/decrypt methods of the circuit/module instead of doing the actual encryption/evaluation/decryption.
    * When this option is set to `True`, encrypt and decrypt are identity functions, and run is a wrapper around simulation. In other words, this option allows to switch off the encryption to quickly test if a function has expected semantic (without paying the price of FHE execution).
  * This is extremely unsafe and should only be used during development.
  * For this reason, it requires **enable\_unsafe\_features** to be set to `True`.
