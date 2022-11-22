# Rounded Table Lookups

{% hint style="warning" %}
Rounded table lookups are only available in [virtual circuits](./virtual_circuits.md) for the time being.
{% endhint %}

Table lookups have a strict constraint on number of bits they support. This can be quite limiting, especially if you don't need the exact precision.

To overcome such shortcomings, rounded table lookup operation is introduced. It's a way to extract most significant bits of a large integer and then applying the table lookup to those bits.

Imagine you have an 8-bit value, but you want to have a 5-bit table lookup, you can call `cnp.round_bit_pattern(input, lsbs_to_remove=3)` and use the value you get in the table lookup.

In Python, evaluation will work like the following:
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

and during homomorphic execution, it'll be converted like this:
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

and then a modified table lookup would be applied to the resulting 5-bits.

Here is a concrete example, let's say you want to apply ReLU to an 18-bit value. Let's see what the original ReLU looks like first:

```python
import matplotlib.pyplot as plt

def relu(x):
    return x if x >= 0 else 0

xs = range(-100_000, 100_000)
ys = [relu(x) for x in xs]

plt.plot(xs, ys)
plt.show()
```

![](../_static/rounded-tlu/relu.png)

Input range is [-100_000, 100_000), which means 18-bit table lookups are required, but they are not supported yet, you can apply rounding operation to the input before passing it to `ReLU` function:

```python
import concrete.numpy as cnp
import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return x if x >= 0 else 0

@cnp.compiler({"x": "encrypted"})
def f(x):
    x = cnp.round_bit_pattern(x, lsbs_to_remove=10)
    return cnp.univariate(relu)(x)

inputset = [-100_000, (100_000 - 1)]
circuit = f.compile(inputset, enable_unsafe_features=True, virtual=True)

xs = range(-100_000, 100_000)
ys = [circuit.encrypt_run_decrypt(x) for x in xs]

plt.plot(xs, ys)
plt.show()
```

in this case we've removed 10 least significant bits of the input and then applied ReLU function to this value to get:

![](../_static/rounded-tlu/10-bits-removed.png)

which is close enough to original ReLU for some cases. If your application is more flexible, you could remove more bits, let's say 12 to get:

![](../_static/rounded-tlu/12-bits-removed.png)

This is very useful, but in some cases, you don't know how many bits your input have, so it's not reliable to specify `lsbs_to_remove` manually. For this reason, `AutoRounder` class is introduced.

```python
import concrete.numpy as cnp
import matplotlib.pyplot as plt
import numpy as np

rounder = cnp.AutoRounder(target_msbs=6)

def relu(x):
    return x if x >= 0 else 0

@cnp.compiler({"x": "encrypted"})
def f(x):
    x = cnp.round_bit_pattern(x, lsbs_to_remove=rounder)
    return cnp.univariate(relu)(x)

inputset = [-100_000, (100_000 - 1)]
cnp.AutoRounder.adjust(f, inputset)  # alternatively, you can use `auto_adjust_rounders=True` below
circuit = f.compile(inputset, enable_unsafe_features=True, virtual=True)

xs = range(-100_000, 100_000)
ys = [circuit.encrypt_run_decrypt(x) for x in xs]

plt.plot(xs, ys)
plt.show()
```

`AutoRounder`s allow you to set how many of the most significant bits to keep, but they need to be adjusted using an inputset to determine how many of the least significant bits to remove. This can be done manually using `cnp.AutoRounder.adjust(function, inputset)`, or by setting `auto_adjust_rounders` to `True` during compilation.

In the example above, `6` of the most significant bits are kept to get:

![](../_static/rounded-tlu/6-bits-kept.png)

You can adjust `target_msbs` depending on your requirements. If you set it to `4` for example, you'd get:

![](../_static/rounded-tlu/4-bits-kept.png)

{% hint style="warning" %}
`AutoRounder`s should be defined outside the function being compiled. They are used to store the result of aqdjustment process, so they shouldn't be created each time the function is called.
{% endhint %}
