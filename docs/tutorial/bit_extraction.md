# Bit Extraction

Some applications require directly manipulating bits of integers. Concrete provides bit extraction operation for such applications.

Bit extraction is capable of extracting a slice of bits from an integer. Index 0 corresponds to the lowest significant bit. The cost of this operation is proportional to the highest significant bit index.

{% hint style="warning" %}
Bit extraction only works in `Native` encoding, which is usually selected when all table lookups in the circuit are below or equal to 8 bits.
{% endhint %}

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.bits(x)[0], fhe.bits(x)[3]

inputset = range(32)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(0b_00000) == (0, 0)
assert circuit.encrypt_run_decrypt(0b_00001) == (1, 0)

assert circuit.encrypt_run_decrypt(0b_01100) == (0, 1)
assert circuit.encrypt_run_decrypt(0b_01101) == (1, 1)
```

Slices can be used for indexing `fhe.bits(value)` as well.

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.bits(x)[1:4]

inputset = range(32)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(0b_01101) == 0b_110
assert circuit.encrypt_run_decrypt(0b_01011) == 0b_101
```

Even slices with negative steps are supported!

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.bits(x)[3:0:-1]

inputset = range(32)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(0b_01101) == 0b_011
assert circuit.encrypt_run_decrypt(0b_01011) == 0b_101
```

Signed integers are supported as well.

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.bits(x)[1:3]

inputset = range(-16, 16)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(-14) == 0b_01  # -14 == 0b_10010 (in two's complement)
assert circuit.encrypt_run_decrypt(-12) == 0b_10  # -12 == 0b_10100 (in two's complement)
```

Lastly, here is a practical use case of bit extraction.

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def is_even(x):
    return 1 - fhe.bits(x)[0]

inputset = [
    np.random.randint(-16, 16, size=(5,))
    for _ in range(100)
]
circuit = is_even.compile(inputset)

sample = np.random.randint(-16, 16, size=(5,))
for value, value_is_even in zip(sample, circuit.encrypt_run_decrypt(sample)):
    print(f"{value} is {'even' if value_is_even else 'odd'}")
```

prints

```
13 is odd
0 is even
-15 is odd
2 is even
-6 is even
```

## Limitations

- Bits cannot be extracted using a negative index.
  - Which means `fhe.bits(x)[-1]` or `fhe.bits(x)[-4:-1]` is not supported for example.
  - The reason for this is we don't know in advance (i.e., before inputset evaluation) how many bits `x` has.
    - For example, let's say you have `x == 10 == 0b_000...0001010`, and you want to do `fhe.bits(x)[-1]`. If the value is 4-bits (i.e., `0b_1010`), the result needs to be `1`, but if it's 6-bits (i.e., `0b_001010`), the result needs to be `0`. Since we don't know the bit-width of `x` before inputset evaluation, we cannot calculate `fhe.bits(x)[-1]`.
  
- When extracting bits using slices in reverse order (i.e., step < 0), start bit **needs** to be provided explicitly.
  - Which means `fhe.bits(x)[::-1]` or `fhe.bits(x)[:2:-1]` is not supported for example.
  - The reason is the same as above.

- When extracting bits of signed values using slices, stop bit **needs** to be provided explicitly.
    - Which means `fhe.bits(x)[1:]` or `fhe.bits(x)[1::2]` is not supported for example.
    - The reason is similar to above.
      - To explain a bit more, signed integers use [two's complement](https://en.wikipedia.org/wiki/Two%27s_complement#:~:text=Two's%20complement%20is%20the%20most,number%20is%20positive%20or%20negative) representation. In this representation, negative values have their most significant bits set to 1 (e.g., `-1 == 0b_11111`, `-2 == 0b_11110`, `-3 == 0b_11101`). Extracting bits always returns a positive value (e.g., `fhe.bits(-1)[1:3] == 0b_11 == 3`) This means if you were to do `fhe.bits(x)[1:]` where `x == -1`, if `x` is 4 bits, the result would be `0b_111 == 7`, but if `x` is 5 bits the result would be `0b_1111 == 15`. Since we don't know the bit-width of `x` before inputset evaluation, we cannot calculate `fhe.bits(x)[1:]`.

- Bits of floats cannot be extracted.
  - Floats are partially supported but extracting their bits is not supported at all.
