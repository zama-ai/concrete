# Rounded Table Lookups

Table lookups have a strict constraint on the number of bits they support. This can limiting, especially if you don't need exact precision.

To overcome this, rounded table lookups are introduced. They provide a way to round the least significant bits of a large integer and then apply the table lookup on the resulting value.

Imagine you have an 8-bit value, but you want to have a 5-bit table lookup. You can call `fhe.round_bit_pattern(input, lsbs_to_remove=3)` and use the value you get in the table lookup.

In Python, evaluation will work like this:

```
0b_0000_0000 => 0b_0000_0000
0b_0000_0001 => 0b_0000_0000
0b_0000_0010 => 0b_0000_0000
0b_0000_0011 => 0b_0000_0000
0b_0000_0100 => 0b_0000_1000
0b_0000_0101 => 0b_0000_1000
0b_0000_0110 => 0b_0000_1000
0b_0000_0111 => 0b_0000_1000

0b_1010_0000 => 0b_1010_0000
0b_1010_0001 => 0b_1010_0000
0b_1010_0010 => 0b_1010_0000
0b_1010_0011 => 0b_1010_0000
0b_1010_0100 => 0b_1010_1000
0b_1010_0101 => 0b_1010_1000
0b_1010_0110 => 0b_1010_1000
0b_1010_0111 => 0b_1010_1000

0b_1010_1000 => 0b_1010_1000
0b_1010_1001 => 0b_1010_1000
0b_1010_1010 => 0b_1010_1000
0b_1010_1011 => 0b_1010_1000
0b_1010_1100 => 0b_1011_0000
0b_1010_1101 => 0b_1011_0000
0b_1010_1110 => 0b_1011_0000
0b_1010_1111 => 0b_1011_0000

0b_1011_1000 => 0b_1011_1000
0b_1011_1001 => 0b_1011_1000
0b_1011_1010 => 0b_1011_1000
0b_1011_1011 => 0b_1011_1000
0b_1011_1100 => 0b_1100_0000
0b_1011_1101 => 0b_1100_0000
0b_1011_1110 => 0b_1100_0000
0b_1011_1111 => 0b_1100_0000
```

During homomorphic execution, it'll be converted like this:

```
0b_0000_0000 => 0b_00000
0b_0000_0001 => 0b_00000
0b_0000_0010 => 0b_00000
0b_0000_0011 => 0b_00000
0b_0000_0100 => 0b_00001
0b_0000_0101 => 0b_00001
0b_0000_0110 => 0b_00001
0b_0000_0111 => 0b_00001

0b_1010_0000 => 0b_10100
0b_1010_0001 => 0b_10100
0b_1010_0010 => 0b_10100
0b_1010_0011 => 0b_10100
0b_1010_0100 => 0b_10101
0b_1010_0101 => 0b_10101
0b_1010_0110 => 0b_10101
0b_1010_0111 => 0b_10101

0b_1010_1000 => 0b_10101
0b_1010_1001 => 0b_10101
0b_1010_1010 => 0b_10101
0b_1010_1011 => 0b_10101
0b_1010_1100 => 0b_10110
0b_1010_1101 => 0b_10110
0b_1010_1110 => 0b_10110
0b_1010_1111 => 0b_10110

0b_1011_1000 => 0b_10111
0b_1011_1001 => 0b_10111
0b_1011_1010 => 0b_10111
0b_1011_1011 => 0b_10111
0b_1011_1100 => 0b_11000
0b_1011_1101 => 0b_11000
0b_1011_1110 => 0b_11000
0b_1011_1111 => 0b_11000
```

{% hint style="info" %}
If kept bits are all zero and removed bits are closer to their biggest possible value than 0, overflows could happen (e.g., `0b_1111_1110` -> `0b_1_0000_0000`. By default, this behavior is detected and prevented by increasing the bit width of the input. However, this introduces some performance overheads.

You can turn this overflow protection off using `fhe.round_bit_pattern(..., overflow_protection=False)`, but beware, you might get unexpected behavior at runtime!
{% endhint %}

Let's say you want to apply ReLU to an 18-bit value. First look at the original ReLU:

```python
import matplotlib.pyplot as plt

def relu(x):
    return x if x >= 0 else 0

xs = range(-100_000, 100_000)
ys = [relu(x) for x in xs]

plt.plot(xs, ys)
plt.show()
```

![](../\_static/rounded-tlu/relu.png)

The input range is \[-100\_000, 100\_000), which means 18-bit table lookups are required, but they are not yet supported. You can apply a rounding operation to the input before passing it to the `ReLU` function:

```python
from concrete import fhe
import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return x if x >= 0 else 0

@fhe.compiler({"x": "encrypted"})
def f(x):
    x = fhe.round_bit_pattern(x, lsbs_to_remove=10)
    return fhe.univariate(relu)(x)

inputset = [-100_000, (100_000 - 1)]
circuit = f.compile(inputset)

xs = range(-100_000, 100_000)
ys = [circuit.simulate(x) for x in xs]

plt.plot(xs, ys)
plt.show()
```

We've removed the 10 least significant bits of the input and then applied the ReLU function to this value to get:

![](../\_static/rounded-tlu/10-bits-removed.png)

This is close enough to original ReLU for some cases. If your application is more flexible, you could remove more bits, let's say 12, to get:

![](../\_static/rounded-tlu/12-bits-removed.png)

This is very useful but, in some cases, you don't know how many bits your input contains, so it's not reliable to specify `lsbs_to_remove` manually. For this reason, `AutoRounder` class is introduced.

```python
from concrete import fhe
import matplotlib.pyplot as plt
import numpy as np

rounder = fhe.AutoRounder(target_msbs=6)

def relu(x):
    return x if x >= 0 else 0

@fhe.compiler({"x": "encrypted"})
def f(x):
    x = fhe.round_bit_pattern(x, lsbs_to_remove=rounder)
    return fhe.univariate(relu)(x)

inputset = [-100_000, (100_000 - 1)]
fhe.AutoRounder.adjust(f, inputset)  # alternatively, you can use `auto_adjust_rounders=True` below
circuit = f.compile(inputset)

xs = range(-100_000, 100_000)
ys = [circuit.simulate(x) for x in xs]

plt.plot(xs, ys)
plt.show()
```

`AutoRounder`s allow you to set how many of the most significant bits to keep, but they need to be adjusted using an inputset to determine how many of the least significant bits to remove. This can be done manually using `fhe.AutoRounder.adjust(function, inputset)`, or by setting `auto_adjust_rounders` to `True` during compilation.

In this case, `6` of the most significant bits are kept to get:

![](../\_static/rounded-tlu/6-bits-kept.png)

You can adjust `target_msbs` depending on your requirements. If you set it to `4`, you get:

![](../\_static/rounded-tlu/4-bits-kept.png)

{% hint style="warning" %}
`AutoRounder`s should be defined outside the function being compiled. They are used to store the result of the adjustment process, so they shouldn't be created each time the function is called. Furthermore, each `AutoRounder` should be used with exactly one `round_bit_pattern` call!
{% endhint %}
