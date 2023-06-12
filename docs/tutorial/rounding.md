# Rounding

Table lookups have a strict constraint on the number of bits they support. This can be limiting, especially if you don't need exact precision.

On top of that, bigger bit-widths are slower in when used in table lookups.

To overcome these, rounded table lookups are introduced. They provide a way to round the least significant bits of a large integer and then apply the table lookup on the resulting value.

Imagine you have an 5-bit value, but you want to have a 3-bit table lookup. You can call `fhe.round_bit_pattern(input, lsbs_to_remove=2)` and use the value you get in the table lookup.

Let's see how rounding works in practice:

```python
import matplotlib.pyplot as plt
import numpy as np
from concrete import fhe

original_bit_width = 5
lsbs_to_remove = 2

assert 0 < lsbs_to_remove < original_bit_width

original_values = list(range(2**original_bit_width))
rounded_values = [
    fhe.round_bit_pattern(value, lsbs_to_remove)
    for value in original_values
]

previous_rounded = rounded_values[0]
for original, rounded in zip(original_values, rounded_values):
    if rounded != previous_rounded:
        previous_rounded = rounded
        print()

    original_binary = np.binary_repr(original, width=(original_bit_width + 1))
    rounded_binary = np.binary_repr(rounded, width=(original_bit_width + 1))

    print(
        f"{original:2} = 0b_{original_binary[:-lsbs_to_remove]}[{original_binary[-lsbs_to_remove:]}] "
        f"=> "
        f"0b_{rounded_binary[:-lsbs_to_remove]}[{rounded_binary[-lsbs_to_remove:]}] = {rounded}"
    )

fig = plt.figure()
ax = fig.add_subplot()

plt.plot(original_values, original_values, label="original", color="black")
plt.plot(original_values, rounded_values, label="rounded", color="green")
plt.legend()

ax.set_aspect("equal", adjustable="box")
plt.show()
```

prints:

```
 0 = 0b_0000[00] => 0b_0000[00] = 0
 1 = 0b_0000[01] => 0b_0000[00] = 0

 2 = 0b_0000[10] => 0b_0001[00] = 4
 3 = 0b_0000[11] => 0b_0001[00] = 4
 4 = 0b_0001[00] => 0b_0001[00] = 4
 5 = 0b_0001[01] => 0b_0001[00] = 4

 6 = 0b_0001[10] => 0b_0010[00] = 8
 7 = 0b_0001[11] => 0b_0010[00] = 8
 8 = 0b_0010[00] => 0b_0010[00] = 8
 9 = 0b_0010[01] => 0b_0010[00] = 8

10 = 0b_0010[10] => 0b_0011[00] = 12
11 = 0b_0010[11] => 0b_0011[00] = 12
12 = 0b_0011[00] => 0b_0011[00] = 12
13 = 0b_0011[01] => 0b_0011[00] = 12

14 = 0b_0011[10] => 0b_0100[00] = 16
15 = 0b_0011[11] => 0b_0100[00] = 16
16 = 0b_0100[00] => 0b_0100[00] = 16
17 = 0b_0100[01] => 0b_0100[00] = 16

18 = 0b_0100[10] => 0b_0101[00] = 20
19 = 0b_0100[11] => 0b_0101[00] = 20
20 = 0b_0101[00] => 0b_0101[00] = 20
21 = 0b_0101[01] => 0b_0101[00] = 20

22 = 0b_0101[10] => 0b_0110[00] = 24
23 = 0b_0101[11] => 0b_0110[00] = 24
24 = 0b_0110[00] => 0b_0110[00] = 24
25 = 0b_0110[01] => 0b_0110[00] = 24

26 = 0b_0110[10] => 0b_0111[00] = 28
27 = 0b_0110[11] => 0b_0111[00] = 28
28 = 0b_0111[00] => 0b_0111[00] = 28
29 = 0b_0111[01] => 0b_0111[00] = 28

30 = 0b_0111[10] => 0b_1000[00] = 32
31 = 0b_0111[11] => 0b_1000[00] = 32
```

and displays:

![](../\_static/rounding/identity.png)

{% hint style="info" %}
If rounded number is one of the last `2**(lsbs_to_remove - 1)` numbers in the input range `[0, 2**original_bit_width)`, an overflow **will** happen.

By default, if overflow is encountered during inputset evaluation, bit-widths will be adjusted accordingly, so it'll be slower, but accurate.

You can turn this overflow protection off for performance using `fhe.round_bit_pattern(..., overflow_protection=False)`, but beware, you might get unexpected behavior at runtime!
{% endhint %}

Now, let's see how rounding can be used in FHE!

```python
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keys",
    single_precision=False,
    parameter_selection_strategy=fhe.ParameterSelectionStrategy.MULTI,
)

input_bit_width = 6
input_range = np.array(range(2**input_bit_width))

timings = {}
results = {}

for lsbs_to_remove in range(input_bit_width):
    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return fhe.round_bit_pattern(x, lsbs_to_remove) ** 2
    
    circuit = f.compile(inputset=[input_range], configuration=configuration)
    circuit.keygen()
    
    encrypted_sample = circuit.encrypt(input_range)
    start = time.time()
    encrypted_result = circuit.run(encrypted_sample)
    end = time.time()
    result = circuit.decrypt(encrypted_result)
    
    took = end - start
    
    timings[lsbs_to_remove] = took
    results[lsbs_to_remove] = result

number_of_figures = len(results)

columns = 1
for i in range(2, number_of_figures):
    if number_of_figures % i == 0:
        columns = i
rows = number_of_figures // columns

fig, axs = plt.subplots(rows, columns)
axs = axs.flatten()

baseline = timings[0]
for lsbs_to_remove in range(input_bit_width):
    timing = timings[lsbs_to_remove]
    speedup = baseline / timing
    print(f"lsbs_to_remove={lsbs_to_remove} => {speedup:.2f}x speedup")

    axs[lsbs_to_remove].set_title(f"lsbs_to_remove={lsbs_to_remove}")
    axs[lsbs_to_remove].plot(input_range, results[lsbs_to_remove])

plt.show()
```

prints:

```
lsbs_to_remove=0 => 1.00x speedup
lsbs_to_remove=1 => 1.20x speedup
lsbs_to_remove=2 => 2.17x speedup
lsbs_to_remove=3 => 3.75x speedup
lsbs_to_remove=4 => 2.64x speedup
lsbs_to_remove=5 => 2.61x speedup
```

{% hint style="info" %}
These can vary from system to system!
{% endhint %}

{% hint style="info" %}
The reason why it doesn't increase every time is that rounding itself has a cost as well. If a lot of bits are removed, rounding itself could take longer than the TLU afterwards!
{% endhint %}

and displays:

![](../\_static/rounding/lsbs_to_remove.png)

{% hint style="info" %}
Feel free to disable overflow protection and see what happens!
{% endhint %}

## Auto Rounders

Rounding is very useful but, in some cases, you don't know how many bits your input contains, so it's not reliable to specify `lsbs_to_remove` manually. For this reason, `AutoRounder` class is introduced.

`AutoRounder` allows you to set how many of the most significant bits to keep, but they need to be adjusted using an inputset to determine how many of the least significant bits to remove. This can be done manually using `fhe.AutoRounder.adjust(function, inputset)`, or by setting `auto_adjust_rounders` configuration to `True` during compilation.

Here is how auto rounders can be used in FHE:

```python
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keys",
    single_precision=False,
    parameter_selection_strategy=fhe.ParameterSelectionStrategy.MULTI,
)

input_bit_width = 6
input_range = np.array(range(2**input_bit_width))

timings = {}
results = {}

for target_msbs in reversed(range(1, input_bit_width + 1)):
    rounder = fhe.AutoRounder(target_msbs)

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return fhe.round_bit_pattern(x, rounder) ** 2

    fhe.AutoRounder.adjust(f, inputset=[input_range])

    circuit = f.compile(inputset=[input_range], configuration=configuration)
    circuit.keygen()

    encrypted_sample = circuit.encrypt(input_range)
    start = time.time()
    encrypted_result = circuit.run(encrypted_sample)
    end = time.time()
    result = circuit.decrypt(encrypted_result)

    took = end - start

    timings[target_msbs] = took
    results[target_msbs] = result

number_of_figures = len(results)

columns = 1
for i in range(2, number_of_figures):
    if number_of_figures % i == 0:
        columns = i
rows = number_of_figures // columns

fig, axs = plt.subplots(rows, columns)
axs = axs.flatten()

baseline = timings[input_bit_width]
for i, target_msbs in enumerate(reversed(range(1, input_bit_width + 1))):
    timing = timings[target_msbs]
    speedup = baseline / timing
    print(f"target_msbs={target_msbs} => {speedup:.2f}x speedup")

    axs[i].set_title(f"target_msbs={target_msbs}")
    axs[i].plot(input_range, results[target_msbs])

plt.show()
```

prints:

```
target_msbs=6 => 1.00x speedup
target_msbs=5 => 1.22x speedup
target_msbs=4 => 1.95x speedup
target_msbs=3 => 3.11x speedup
target_msbs=2 => 2.23x speedup
target_msbs=1 => 2.34x speedup
```

and displays:

![](../\_static/rounding/msbs_to_keep.png)

{% hint style="warning" %}
`AutoRounder`s should be defined outside the function that is being compiled. They are used to store the result of the adjustment process, so they shouldn't be created each time the function is called. Furthermore, each `AutoRounder` should be used with exactly one `round_bit_pattern` call!
{% endhint %}
