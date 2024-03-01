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
  * Error probability for individual table lookups. If set, all table lookups will have the probability of a non-exact result smaller than the set value. See [Exactness](../tutorial/table\_lookups.md#table-lookup-exactness) to learn more.
* **global\_p\_error**: Optional\[float] = None
  * Global error probability for the whole circuit. If set, the whole circuit will have the probability of a non-exact result smaller than the set value. See [Exactness](../tutorial/table\_lookups.md#table-lookup-exactness) to learn more.
* **single\_precision**: bool = False
  * Use single precision for the whole circuit.
* **parameter\_selection\_strategy**: (fhe.ParameterSelectionStrategy) = fhe.ParameterSelectionStrategy.MULTI
  * Set how cryptographic parameters are selected.
* **multi\_parameter\_strategy**: fhe.MultiParameterStrategy = fhe.MultiParameterStrategy.PRECISION
  * Set the level of circuit partionning when using `fhe.ParameterSelectionStrategy.MULTI`.
  * `PRECISION`: all TLU with same input precision have their own parameters.
  * `PRECISION_AND_NORM2`: all TLU with same input precision and output [norm2](../../compilers/concrete-optimizer/v0-parameters/README.md) have their own parameters.
* **loop\_parallelize**: bool = True
  * Enable loop parallelization in the compiler.
* **dataflow\_parallelize**: bool = False
  * Enable dataflow parallelization in the compiler.
* **auto\_parallelize**: bool = False
  * Enable auto parallelization in the compiler.
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
* **progress\_tag**: Union[bool, int] = False,
  * How many nested tag elements to display with the progress bar. `True` means all tag elements and `False` disables the display. `2` will display `elmt1.elmt2`
* **fhe\_simulation**: bool = False
  * Enable FHE simulation. Can be enabled later using `circuit.enable_fhe_simulation()`.
* **fhe\_execution**: bool = True
  * Enable FHE execution. Can be enabled later using `circuit.enable_fhe_execution()`.
* **compiler_debug_mode**: bool = False,
  * Enable/disable debug mode of the compiler. This can show a lot of information, including passes and pattern rewrites.
* **compiler_verbose_mode**: bool = False,
  * Enable/disable verbose mode of the compiler. This mainly shows logs from the compiler, and is less verbose than the debug mode.
* **comparison_strategy_preference**: Optional[Union[ComparisonStrategy, str, List[Union[ComparisonStrategy, str]]]] = None
  * Specify preference for comparison strategies, can be a single strategy or an ordered list of strategies. See [Comparisons](../tutorial/comparisons.md) to learn more.
* **bitwise_strategy_preference**: Optional[Union[BitwiseStrategy, str, List[Union[BitwiseStrategy, str]]]] = None
  * Specify preference for bitwise strategies, can be a single strategy or an ordered list of strategies. See [Bitwise](../tutorial/bitwise.md) to learn more.
* **shifts_with_promotion**: bool = True,
  * Enable promotions in encrypted shifts instead of casting in runtime. See [Bitwise#Shifts](../tutorial/bitwise.md#Shifts) to learn more.
* **composable**: bool = False,
  * Specify that the function must be composable with itself.
* **relu_on_bits_threshold**: int = 7,
  * Bit-width to start implementing the ReLU extension with [fhe.bits](../tutorial/bit_extraction.md).
* **relu_on_bits_chunk_size**: int = 3,
  * Chunk size of the ReLU extension when [fhe.bits](../tutorial/bit_extraction.md) implementation is used.
* **if_then_else_chunk_size**: int = 3
  * Chunk size to use when converting `fhe.if_then_else` extension.
* **rounding_exactness** : Exactness = `fhe.Exactness.EXACT`
  * Set default exactness mode for the rounding operation:
  * `EXACT`: threshold for rounding up or down is exactly centered between upper and lower value,
  * `APPROXIMATE`: faster but threshold for rounding up or down is approximately centered with pseudo-random shift.
  * Precise and more complete behavior is described in  [fhe.rounding_bit_pattern](../tutorial/rounding.md).
* **approximate_rounding_config** : ApproximateRoundingConfig = `fhe.ApproximateRoundingConfig()`:
  * Provide more fine control on [approximate rounding](../tutorial/rounding.md#approximate-rounding-features):
  * to enable exact cliping,
  * or/and approximate clipping which make overflow protection faster.
