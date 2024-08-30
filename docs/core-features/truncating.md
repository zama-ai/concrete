# Truncating

This document details the concept of truncating, and how it is used in Concrete to make some FHE computations especially faster.

Table lookups have a strict constraint on the number of bits they support. This can be limiting, especially if you don't need exact precision. As well as this, using larger bit-widths leads to slower table lookups.

To overcome these issues, truncated table lookups are introduced. This operation provides a way to zero the least significant bits of a large integer and then apply the table lookup on the resulting (smaller) value.

Imagine you have a 5-bit value, you can use `fhe.truncate_bit_pattern(value, lsbs_to_remove=2)` to truncate it (here the last 2 bits are discarded). Once truncated, value will remain in 5-bits (e.g., 22 = 0b10110 would be truncated to 20 = 0b10100), and the last 2 bits of it would be zero. Concrete uses this to optimize table lookups on the truncated value, the 5-bit table lookup gets optimized to a 3-bit table lookup, which is much faster!

Let's see how truncation works in practice:

```python
import matplotlib.pyplot as plt
import numpy as np
from concrete import fhe

original_bit_width = 5
lsbs_to_remove = 2

assert 0 < lsbs_to_remove < original_bit_width

original_values = list(range(2**original_bit_width))
truncated_values = [
    fhe.truncate_bit_pattern(value, lsbs_to_remove)
    for value in original_values
]

previous_truncated = truncated_values[0]
for original, truncated in zip(original_values, truncated_values):
    if truncated != previous_truncated:
        previous_truncated = truncated
        print()

    original_binary = np.binary_repr(original, width=(original_bit_width + 1))
    truncated_binary = np.binary_repr(truncated, width=(original_bit_width + 1))

    print(
        f"{original:2} = 0b_{original_binary[:-lsbs_to_remove]}[{original_binary[-lsbs_to_remove:]}] "
        f"=> "
        f"0b_{truncated_binary[:-lsbs_to_remove]}[{truncated_binary[-lsbs_to_remove:]}] = {truncated}"
    )

fig = plt.figure()
ax = fig.add_subplot()

plt.plot(original_values, original_values, label="original", color="black")
plt.plot(original_values, truncated_values, label="truncated", color="green")
plt.legend()

ax.set_aspect("equal", adjustable="box")
plt.show()
```

prints:

```
 0 = 0b_0000[00] => 0b_0000[00] = 0
 1 = 0b_0000[01] => 0b_0000[00] = 0
 2 = 0b_0000[10] => 0b_0000[00] = 0
 3 = 0b_0000[11] => 0b_0000[00] = 0

 4 = 0b_0001[00] => 0b_0001[00] = 4
 5 = 0b_0001[01] => 0b_0001[00] = 4
 6 = 0b_0001[10] => 0b_0001[00] = 4
 7 = 0b_0001[11] => 0b_0001[00] = 4

 8 = 0b_0010[00] => 0b_0010[00] = 8
 9 = 0b_0010[01] => 0b_0010[00] = 8
10 = 0b_0010[10] => 0b_0010[00] = 8
11 = 0b_0010[11] => 0b_0010[00] = 8

12 = 0b_0011[00] => 0b_0011[00] = 12
13 = 0b_0011[01] => 0b_0011[00] = 12
14 = 0b_0011[10] => 0b_0011[00] = 12
15 = 0b_0011[11] => 0b_0011[00] = 12

16 = 0b_0100[00] => 0b_0100[00] = 16
17 = 0b_0100[01] => 0b_0100[00] = 16
18 = 0b_0100[10] => 0b_0100[00] = 16
19 = 0b_0100[11] => 0b_0100[00] = 16

20 = 0b_0101[00] => 0b_0101[00] = 20
21 = 0b_0101[01] => 0b_0101[00] = 20
22 = 0b_0101[10] => 0b_0101[00] = 20
23 = 0b_0101[11] => 0b_0101[00] = 20

24 = 0b_0110[00] => 0b_0110[00] = 24
25 = 0b_0110[01] => 0b_0110[00] = 24
26 = 0b_0110[10] => 0b_0110[00] = 24
27 = 0b_0110[11] => 0b_0110[00] = 24

28 = 0b_0111[00] => 0b_0111[00] = 28
29 = 0b_0111[01] => 0b_0111[00] = 28
30 = 0b_0111[10] => 0b_0111[00] = 28
31 = 0b_0111[11] => 0b_0111[00] = 28
```

and displays:

![](../\_static/truncating/identity.png)

Now, let's see how truncating can be used in FHE.

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
)

input_bit_width = 6
input_range = np.array(range(2**input_bit_width))

timings = {}
results = {}

for lsbs_to_remove in range(input_bit_width):
    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return fhe.truncate_bit_pattern(x, lsbs_to_remove) ** 2
    
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
lsbs_to_remove=1 => 1.69x speedup
lsbs_to_remove=2 => 3.48x speedup
lsbs_to_remove=3 => 3.06x speedup
lsbs_to_remove=4 => 3.46x speedup
lsbs_to_remove=5 => 3.14x speedup
```

{% hint style="info" %}
These speed-ups can vary from system to system.
{% endhint %}

{% hint style="info" %}
The reason why the speed-up is not increasing with `lsbs_to_remove` is because the truncating operation itself has a cost: each bit removal is a PBS. Therefore, if a lot of bits are removed, truncation itself could take longer than the bigger TLU which is evaluated afterwards.
{% endhint %}

and displays:

![](../\_static/truncating/lsbs_to_remove.png)

## Auto Truncators

Truncating is very useful but, in some cases, you don't know how many bits your input contains, so it's not reliable to specify `lsbs_to_remove` manually. For this reason, the `AutoTruncator` class is introduced.

`AutoTruncator` allows you to set how many of the most significant bits to keep, but they need to be adjusted using an inputset to determine how many of the least significant bits to remove. This can be done manually using `fhe.AutoTruncator.adjust(function, inputset)`, or by setting `auto_adjust_truncators` configuration to `True` during compilation.

Here is how auto truncators can be used in FHE:

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
    truncator = fhe.AutoTruncator(target_msbs)

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return fhe.truncate_bit_pattern(x, lsbs_to_remove=truncator) ** 2

    fhe.AutoTruncator.adjust(f, inputset=[input_range])

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
target_msbs=5 => 1.80x speedup
target_msbs=4 => 3.47x speedup
target_msbs=3 => 3.02x speedup
target_msbs=2 => 3.38x speedup
target_msbs=1 => 3.37x speedup
```

and displays:

![](../\_static/truncating/msbs_to_keep.png)

{% hint style="warning" %}
`AutoTruncator`s should be defined outside the function that is being compiled. They are used to store the result of the adjustment process, so they shouldn't be created each time the function is called. Furthermore, each `AutoTruncator` should be used with exactly one `truncate_bit_pattern` call.
{% endhint %}
