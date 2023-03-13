# Configure

The behavior of **Concrete** can be customized using `Configuration`s:

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

Alternatively, you can overwrite individual options as kwargs to `compile` method:

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
Additional kwarg to `compile` function have higher precedence. So if you set an option in both `configuration` and in `compile` methods, the value in the `compile` method will be used.
{% endhint %}

## Options

* **show\_graph**: Optional[bool] = None
  * Whether to print computation graph during compilation.
    `True` means always to print, `False` means always to not print, `None` means print depending on verbose configuration below.

* **show\_mlir**: Optional[bool] = None
  * Whether to print MLIR during compilation.
    `True` means always to print, `False` means always to not print, `None` means print depending on verbose configuration below.

* **show\_optimizer**: Optional[bool] = None
  * Whether to print optimizer output during compilation.
    `True` means always to print, `False` means always to not print, `None` means print depending on verbose configuration below.

* **verbose**: bool = False
  * Whether to print details related to compilation.

* **dump\_artifacts\_on\_unexpected\_failures**: bool = True
  * Whether to export debugging artifacts automatically on compilation failures.

* **auto\_adjust\_rounders**: bool = False
    * Whether to adjust rounders automatically.

* **p\_error**: Optional[float] = None
  * Error probability for individual table lookups. If set, all table lookups will have the probability of non-exact result smaller than the set value. See [Exactness](../getting-started/exactness.md) to learn more.

* **global\_p\_error**: Optional[float] = None
  * Global error probability for the whole circuit. If set, the whole circuit will have the probability of non-exact result smaller than the set value. See [Exactness](../getting-started/exactness.md) to learn more.

* **single_precision**: bool = True
  * Whether to use single precision for the whole circuit.

* **jit**: bool = False
  * Whether to use JIT compilation.

* **loop\_parallelize**: bool = True
  * Whether to enable loop parallelization in the compiler.

* **dataflow\_parallelize**: bool = False
  * Whether to enable dataflow parallelization in the compiler.

* **auto\_parallelize**: bool = False
  * Whether to enable auto parallelization in the compiler.

* **enable\_unsafe\_features**: bool = False
  * Whether to enable unsafe features.

* **use\_insecure\_key\_cache**: bool = False _(Unsafe)_
  * Whether to use the insecure key cache.

* **insecure\_key\_cache\_location**: Optional\[Union\[Path, str]] = None
  * Location of insecure key cache.
