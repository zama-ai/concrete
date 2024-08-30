# Direct circuits

This document explains the concept of direct circuits in Concrete, which is another way to compile circuit without having to give a proper inputset.

{% hint style="warning" %}
Direct circuits are still experimental. It is very easy to make mistakes (e.g., due to no overflow checks or type coercion) while using direct circuits, so utilize them with care.
{% endhint %}

For some applications, the data types of inputs, intermediate values, and outputs are known (e.g., for manipulating bytes, you would want to use uint8). Using inputsets to determine bounds in these cases is not necessary, and can even be error-prone. Therefore, another interface for defining such circuits is introduced:

```python
from concrete import fhe

@fhe.circuit({"x": "encrypted"})
def circuit(x: fhe.uint8):
    return x + 42

assert circuit.encrypt_run_decrypt(10) == 52
```

There are a few differences between direct circuits and traditional circuits:

* Remember that the resulting dtype for each operation will be determined by its inputs. This can lead to some unexpected results if you're not careful (e.g., if you do `-x` where `x: fhe.uint8`, you won't receive a negative value as the result will be `fhe.uint8` as well)
* There is no inputset evaluation when using fhe types in `.astype(...)` calls (e.g., `np.sqrt(x).astype(fhe.uint4)`), so the bit width of the output cannot be determined.
* Specify the resulting data type in [univariate](../core-features/extensions.md#fheunivariatefunction) extension (e.g., `fhe.univariate(function, outputs=fhe.uint4)(x)`), for the same reason as above.
* Be careful with overflows. With inputset evaluation, you'll get bigger bit widths but no overflows. With direct definition, you must ensure that there aren't any overflows manually.

Let's review a more complicated example to see how direct circuits behave:

```python
from concrete import fhe
import numpy as np

def square(value):
    return value ** 2

@fhe.circuit({"x": "encrypted", "y": "encrypted"})
def circuit(x: fhe.uint8, y: fhe.int2):
    a = x + 10
    b = y + 10

    c = np.sqrt(a).round().astype(fhe.uint4)
    d = fhe.univariate(square, outputs=fhe.uint8)(b)

    return d - c

print(circuit)
```

This prints:

```
%0 = x                       # EncryptedScalar<uint8>
%1 = y                       # EncryptedScalar<int2>
%2 = 10                      # ClearScalar<uint4>
%3 = add(%0, %2)             # EncryptedScalar<uint8>
%4 = 10                      # ClearScalar<uint4>
%5 = add(%1, %4)             # EncryptedScalar<int4>
%6 = subgraph(%3)            # EncryptedScalar<uint4>
%7 = square(%5)              # EncryptedScalar<uint8>
%8 = subtract(%7, %6)        # EncryptedScalar<uint8>
return %8

Subgraphs:

    %6 = subgraph(%3):

        %0 = input                         # EncryptedScalar<uint8>
        %1 = sqrt(%0)                      # EncryptedScalar<float64>
        %2 = around(%1, decimals=0)        # EncryptedScalar<float64>
        %3 = astype(%2)                    # EncryptedScalar<uint4>
        return %3
```

Here is the breakdown of the assigned data types:

```
%0 is uint8 because it's specified in the definition
%1 is  int2 because it's specified in the definition
%2 is uint4 because it's the constant 10
%3 is uint8 because it's the addition between uint8 and uint4
%4 is uint4 because it's the constant 10
%5 is  int4 because it's the addition between int2 and uint4
%6 is uint4 because it's specified in astype
%7 is uint8 because it's specified in univariate
%8 is uint8 because it's subtraction between uint8 and uint4
```

As you can see, `%8` is subtraction of two unsigned values, and the result is unsigned as well. In the case that `c > d`, we have an overflow, and this results in undefined behavior.
