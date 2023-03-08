# Direct Circuits

{% hint style="warning" %}
Direct circuits are still experimental, and it's very easy to shoot yourself in the foot (e.g., no overflow checks, no type coercion) while using them so utilize them with care.
{% endhint %}

For some applications, data types of inputs, intermediate values and outputs are known (e.g., for manipulating bytes, you would want to use uint8). For such cases, using inputsets to determine bounds are not necessary, or even error-prone. Therefore, another interface for defining such circuits, is introduced:

```python
import concrete.numpy as cnp

@cnp.circuit({"x": "encrypted"})
def circuit(x: cnp.uint8):
    return x + 42

assert circuit.encrypt_run_decrypt(10) == 52
```

There are a few differences between direct circuits and traditional circuits though:

- You need to remember that resulting dtype for each operation will be determined by its inputs. This can lead to some unexpected results if you're not careful (e.g., if you do `-x` where `x: cnp.uint8`, you'll not get the negative value as the result will be `cnp.uint8` as well)
- You need to use cnp types in `.astype(...)` calls (e.g., `np.sqrt(x).astype(cnp.uint4)`). This is because there are no inputset evaluation, so cannot determine the bit-width of the output.
- You need to specify the resulting data type in [univariate](./extensions.md#cnpunivariatefunction) extension (e.g., `cnp.univariate(function, outputs=cnp.uint4)(x)`), because of the same reason as above.
- You need to be careful with overflows. With inputset evaluation, you'll get bigger bit-widths but no overflows, with direct definition, you're responsible to ensure there aren't any overflows!

Let's go over a more complicated example to see how direct circuits behave:

```python
import concrete.numpy as cnp
import numpy as np

def square(value):
    return value ** 2

@cnp.circuit({"x": "encrypted", "y": "encrypted"})
def circuit(x: cnp.uint8, y: cnp.int2):
    a = x + 10
    b = y + 10

    c = np.sqrt(a).round().astype(cnp.uint4)
    d = cnp.univariate(square, outputs=cnp.uint8)(b)

    return d - c

print(circuit)
```
prints
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
And here is the breakdown of assigned data types:
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

As you can see, `%8` is subtraction of two unsigned values, and it's unsigned as well. In an overflow condition where `c > d`, it'll result in undefined behavior.
