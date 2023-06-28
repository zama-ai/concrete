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
Additional kwarg to `compile` functions take higher precedence. So if you set the option in both `configuration` and `compile` methods, the value in the `compile` method will be used.
{% endhint %}

## Options

* **show\_graph**: Optional\[bool] = None
  * Print computation graph during compilation. `True` means always print, `False` means never print, `None` means print depending on verbose configuration below.
* **show\_mlir**: Optional\[bool] = None
  * Print MLIR during compilation. `True` means always print, `False` means never print, `None` means print depending on verbose configuration below.
* **show\_optimizer**: Optional\[bool] = None
  * Print optimizer output during compilation. `True` means always print, `False` means never print, `None` means print depending on verbose configuration below.
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
* **single\_precision**: bool = True
  * Use single precision for the whole circuit.
* **jit**: bool = False
  * Enable JIT compilation.
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
