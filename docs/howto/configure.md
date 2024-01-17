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
  * Error probability for individual table lookups. If set, all table lookups will have the probability of a non-exact result smaller than the set value. See [Exactness](../getting-started/exactness.md) to learn more.
* **global\_p\_error**: Optional\[float] = None
  * Global error probability for the whole circuit. If set, the whole circuit will have the probability of a non-exact result smaller than the set value. See [Exactness](../getting-started/exactness.md) to learn more.
* **single\_precision**: bool = False
  * Use single precision for the whole circuit.
* **parameter\_selection\_strategy**: (fhe.ParameterSelectionStrategy) = fhe.ParameterSelectionStrategy.MULTI
  * Set how cryptographic parameters are selected.
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
