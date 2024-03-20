# Composition

`concrete-python` supports circuit __composition__, which allows the output of a circuit execution to be used directly as an input without decryption. We can execute the circuit as many time as we want by forwarding outputs without decrypting intermediate values. This feature enables a new range of applications, including support for control flow in pure (cleartext) python.

Here is a first simple example that uses composition to implement a simple counter in FHE:

```python
from concrete import fhe

@fhe.compiler({"counter": "encrypted"})
def increment(counter):
   return (counter + 1) % 100

print("Compiling `increment` function")
increment_fhe = increment.compile(list(range(0, 100)), composable=True)

print("Generating keyset ...")
increment_fhe.keygen()

print("Encrypting the initial counter value")
counter = 0
counter_enc = increment_fhe.encrypt(counter)

print(f"| iteration || decrypted | cleartext |")
for i in range(10):
    counter_enc = increment_fhe.run(counter_enc)
    counter = increment(counter)

    # For demo purpose; no decryption is needed.
    counter_dec = increment_fhe.decrypt(counter_enc)
    print(f"|     {i}     || {counter_dec:<9} | {counter:<9} |")
```

Note the use of the `composable` flag in the `compile` call. It instructs the compiler to ensure the circuit can be called on its own outputs (see [Limitations section](#limitations) for more details). Executing this script should give the following output:

```shell
Compiling `increment` function
Generating keyset ...
Encrypting the initial counter value
| iteration || decrypted | cleartext |
|     0     || 1         | 1         |
|     1     || 2         | 2         |
|     2     || 3         | 3         |
|     3     || 4         | 4         |
|     4     || 5         | 5         |
|     5     || 6         | 6         |
|     6     || 7         | 7         |
|     7     || 8         | 8         |
|     8     || 9         | 9         |
|     9     || 10        | 10        |
```

## Multi inputs, multi outputs

Composition is not limited to 1-to-1 circuits, it can also be used with circuits with multiple inputs and multiple outputs. Here is an example that computes the 10 first elements of the Fibonacci sequence in FHE:

```python
from concrete import fhe

def noise_reset(x):
   return fhe.univariate(lambda x: x)(x)

@fhe.compiler({"n1th": "encrypted", "nth": "encrypted"})
def fib(n1th, nth):
   return noise_reset(nth), noise_reset(n1th + nth)

print("Compiling `fib` function ...")
inputset = list(zip(range(0, 100), range(0, 100)))
fib_fhe = fib.compile(inputset, composable=True)

print("Generating keyset ...")
fib_fhe.keygen()

print("Encrypting initial values")
n1th = 1
nth = 2
(n1th_enc, nth_enc) = fib_fhe.encrypt(n1th, nth)

print(f"|           ||        (n-1)-th       |         n-th          |")
print(f"| iteration || decrypted | cleartext | decrypted | cleartext |")
for i in range(10):
   (n1th_enc, nth_enc) = fib_fhe.run(n1th_enc, nth_enc)
   (n1th, nth) = fib(n1th, nth)
   
    # For demo purpose; no decryption is needed.
   (n1th_dec, nth_dec) = fib_fhe.decrypt(n1th_enc, nth_enc)
   print(f"|     {i}     || {n1th_dec:<9} | {n1th:<9} | {nth_dec:<9} | {nth:<9} |")
```

Executing this script will provide the following output:

```shell
Compiling `fib` function ...
Generating keyset ...
Encrypting initial values
|           ||        (n-1)-th       |         n-th          |
| iteration || decrypted | cleartext | decrypted | cleartext |
|     0     || 2         | 2         | 3         | 3         |
|     1     || 3         | 3         | 5         | 5         |
|     2     || 5         | 5         | 8         | 8         |
|     3     || 8         | 8         | 13        | 13        |
|     4     || 13        | 13        | 21        | 21        |
|     5     || 21        | 21        | 34        | 34        |
|     6     || 34        | 34        | 55        | 55        |
|     7     || 55        | 55        | 89        | 89        |
|     8     || 89        | 89        | 144       | 144       |
|     9     || 144       | 144       | 233       | 233       |
```

Though it is not visible in this example, there is no limitations on the number of inputs and outputs. There is also no need for a specific logic regarding how we forward values from outputs to inputs; those could be switched for instance. 

{% hint style="info" %}
See below in the [Limitations section](#limitations), for explanations about the use of `noise_reset`.
{% endhint %}

## Iteration support

With the previous example we see that to some extent, composition allows to support iteration with cleartext iterands. That is, loops with the following shape :

```python
for i in some_cleartext_constant_range:
    # Do something in FHE in the loop body, implement as an FHE circuit.
```

With this pattern, we can also support unbounded loops or complex dynamic condition, as long as this condition is computed in pure cleartext python. Here is an example that computes the [Collatz sequence](https://en.wikipedia.org/wiki/Collatz_conjecture):

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def collatz(x):

    y = x // 2
    z = 3 * x + 1

    is_x_odd = fhe.bits(x)[0]

    # In a fast way, compute ans = is_x_odd * (z - y) + y
    ans = fhe.multivariate(lambda b, x: b * x)(is_x_odd, z - y) + y

    is_one = ans == 1

    return ans, is_one


print("Compiling `collatz` function ...")
inputset = [i for i in range(63)]
collatz_fhe = collatz.compile(inputset, composable=True)

print("Generating keyset ...")
collatz_fhe.keygen()

print("Encrypting initial value")
x = 19
x_enc = collatz_fhe.encrypt(x)
is_one_enc = None

print(f"| decrypted | cleartext |")
while is_one_enc is None or not collatz_fhe.decrypt(is_one_enc):
    x_enc, is_one_enc = collatz_fhe.run(x_enc)
    x, is_one = collatz(x)

    # For demo purpose; no decryption is needed.
    x_dec = collatz_fhe.decrypt(x_enc)
    print(f"| {x_dec:<9} | {x:<9} |")
```

Which prints:

```shell
Compiling `collatz` function ...
Generating keyset ...
Encrypting initial value
| decrypted | cleartext |
| 58        | 58        |
| 29        | 29        |
| 88        | 88        |
| 44        | 44        |
| 22        | 22        |
| 11        | 11        |
| 34        | 34        |
| 17        | 17        |
| 52        | 52        |
| 26        | 26        |
| 13        | 13        |
| 40        | 40        |
| 20        | 20        |
| 10        | 10        |
| 5         | 5         |
| 16        | 16        |
| 8         | 8         |
| 4         | 4         |
| 2         | 2         |
| 1         | 1         |
```

Here we use a while loop that keeps iterating as long as the decryption of the running value is different from `1`. Again, the loop body is implemented in FHE, but the iteration control has to be in the clear.

## Limitations

Depending on the circuit, supporting composition may add a non-negligible overhead when compared to a non-composable version. Indeed, to be composable a circuit must verify two conditions:
1) All inputs and outputs must share the same precision and the same crypto-parameters: the most expensive parameters that would otherwise be used for a single input or output, are generalized to all inputs and outputs.
2) There must be a noise refresh in every path between an input and an output: some circuits will need extra PBSes to be added to allow composability.

The first point is handled automatically by the compiler, no change to the circuit is needed to ensure the right precisions are used. 

For the second point, since adding a PBS has an impact on performance, we do not ade them on behalf of the user. For instance, to implement a circuit that doubles an encrypted value, we would write something like:

```python
@fhe.compiler({"counter": "encrypted"})
def double(counter):
   return counter * 2
```

This is a valid circuit when `composable` is not used, but when compiled with composition activated, a `RuntimeError: Program can not be composed: ...` error is reported, signalling that an extra PBS must be added. To solve this situation, and turn this circuit into a composable one, one can use the following snippet to add a PBS at the end of your circuit:

```python
def noise_reset(x):
   return fhe.univariate(lambda x: x)(x)

@fhe.compiler({"counter": "encrypted"})
def double(counter):
   return noise_reset(counter * 2)
```
