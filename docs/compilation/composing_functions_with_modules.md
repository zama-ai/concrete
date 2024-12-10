# Composing functions with modules

This document explains how to compile Fully Homomorphic Encryption (FHE) modules containing multiple functions using **Concrete**.

Deploying a server that contains many compatible functions is important for some use cases. With **Concrete**, you can compile FHE modules containing as many functions as needed.

These modules support the composition of different functions, meaning that the encrypted result of one function can be used as the input for another function without needing to decrypt it first. Additionally, a module is [deployed in a single artifact](../guides/deploy.md#deployment-of-modules), making it as simple to use as a single-function project.

## Single inputs / outputs

The following example demonstrates how to create an FHE module:
```python
from concrete import fhe

@fhe.module()
class Counter:
    @fhe.function({"x": "encrypted"})
    def inc(x):
        return (x + 1) % 20

    @fhe.function({"x": "encrypted"})
    def dec(x):
        return (x - 1) % 20
```

Then, to compile the `Counter` module, use the `compile` method with a dictionary of input-sets for each function:

```python
inputset = list(range(20))
CounterFhe = Counter.compile({"inc": inputset, "dec": inputset})
```

After the module is compiled, you can encrypt and call the different functions as follows:

```python
x = 5
x_enc = CounterFhe.inc.encrypt(x)
x_inc_enc = CounterFhe.inc.run(x_enc)
x_inc = CounterFhe.inc.decrypt(x_inc_enc)
assert x_inc == 6

x_inc_dec_enc = CounterFhe.dec.run(x_inc_enc)
x_inc_dec = CounterFhe.dec.decrypt(x_inc_dec_enc)
assert x_inc_dec == 5

for _ in range(10):
    x_enc = CounterFhe.inc.run(x_enc)
x_dec = CounterFhe.inc.decrypt(x_enc)
assert x_dec == 15
```

You can generate the keyset beforehand by calling `keygen()` method on the compiled module:

```python
CounterFhe.keygen()
```

## Multi inputs / outputs

Composition is not limited to single input / single output. Here is an example that computes the 10 first elements of the Fibonacci sequence in FHE:

```python
from concrete import fhe

def noise_reset(x):
   return fhe.univariate(lambda x: x)(x)

@fhe.module()
class Fibonacci:

    @fhe.function({"n1th": "encrypted", "nth": "encrypted"})
    def fib(n1th, nth):
       return noise_reset(nth), noise_reset(n1th + nth)

print("Compiling `Fibonacci` module ...")
inputset = list(zip(range(0, 100), range(0, 100)))
FibonacciFhe = Fibonacci.compile({"fib": inputset})

print("Generating keyset ...")
FibonacciFhe.keygen()

print("Encrypting initial values")
n1th = 1
nth = 2
(n1th_enc, nth_enc) = FibonacciFhe.fib.encrypt(n1th, nth)

print(f"|           ||        (n-1)-th       |         n-th          |")
print(f"| iteration || decrypted | cleartext | decrypted | cleartext |")
for i in range(10):
   (n1th_enc, nth_enc) = FibonacciFhe.fib.run(n1th_enc, nth_enc)
   (n1th, nth) = Fibonacci.fib(n1th, nth)

    # For demo purpose; no decryption is needed.
   (n1th_dec, nth_dec) = FibonacciFhe.fib.decrypt(n1th_enc, nth_enc)
   print(f"|     {i}     || {n1th_dec:<9} | {n1th:<9} | {nth_dec:<9} | {nth:<9} |")
```

Executing this script will provide the following output:

```shell
Compiling `Fibonacci` module ...
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

## Iterations

Modules support iteration with cleartext iterands to some extent, particularly for loops structured like this:

```python
for i in some_cleartext_constant_range:
    # Do something in FHE in the loop body, implemented as an FHE function.
```
Unbounded loops or complex dynamic conditions are also supported, as long as these conditions are computed in pure cleartext in Python. The following example computes the [Collatz sequence](https://en.wikipedia.org/wiki/Collatz_conjecture):

```python
from concrete import fhe

@fhe.module()
class Collatz:

    @fhe.function({"x": "encrypted"})
    def collatz(x):

        y = x // 2
        z = 3 * x + 1

        is_x_odd = fhe.bits(x)[0]

        # In a fast way, compute ans = is_x_odd * (z - y) + y
        ans = fhe.multivariate(lambda b, x: b * x)(is_x_odd, z - y) + y

        is_one = ans == 1

        return ans, is_one


print("Compiling `Collatz` module ...")
inputset = [i for i in range(63)]
CollatzFhe = Collatz.compile({"collatz": inputset})

print("Generating keyset ...")
CollatzFhe.keygen()

print("Encrypting initial value")
x = 19
x_enc = CollatzFhe.collatz.encrypt(x)
is_one_enc = None

print(f"| decrypted | cleartext |")
while is_one_enc is None or not CollatzFhe.collatz.decrypt(is_one_enc):
    x_enc, is_one_enc = CollatzFhe.collatz.run(x_enc)
    x, is_one = Collatz.collatz(x)

    # For demo purpose; no decryption is needed.
    x_dec = CollatzFhe.collatz.decrypt(x_enc)
    print(f"| {x_dec:<9} | {x:<9} |")
```

This script prints the following output:

```shell
Compiling `Collatz` module ...
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
In this example, a while loop iterates until the decrypted value equals 1. The loop body is implemented in FHE, but the iteration control must be in cleartext.

## Runtime optimization

By default, when using modules, all inputs and outputs of every function are compatible, sharing the same precision and crypto-parameters. This approach applies the crypto-parameters of the most costly code path to all code paths. This simplicity may be costly and unnecessary for some use cases.

To optimize runtime, we provide finer-grained control over the composition policy via the `composition` module attribute. Here is an example:

```python
from concrete import fhe

@fhe.module()
class Collatz:

    @fhe.function({"x": "encrypted"})
    def collatz(x):
        y = x // 2
        z = 3 * x + 1
        is_x_odd = fhe.bits(x)[0]
        ans = fhe.multivariate(lambda b, x: b * x)(is_x_odd, z - y) + y
        is_one = ans == 1
        return ans, is_one

    composition = fhe.AllComposable()
```

You have 3 options for the `composition` attribute:

1. **`fhe.AllComposable` (default)**:  This policy ensures that all ciphertexts used in the module are compatible. It is the least restrictive policy but the most costly in terms of performance.

2. **`fhe.NotComposable`**: This policy is the most restrictive but the least costly. It is suitable when you do not need any composition and only want to pack multiple functions in a single artifact.

3. **`fhe.Wired`**: This policy allows you to define custom composition rules. You can specify which outputs of a function can be forwarded to which inputs of another function.

Note that, in case of complex composition logic another option is to rely on [[composing_functions_with_modules#Automatic module tracing]] to automatically derive the composition from examples.

    Here is an example:
```python
from concrete import fhe
from fhe import Wired, Wire, Output, Input

@fhe.module()
class Collatz:

    @fhe.function({"x": "encrypted"})
    def collatz(x):
        y = x // 2
        z = 3 * x + 1
        is_x_odd = fhe.bits(x)[0]
        ans = fhe.multivariate(lambda b, x: b * x)(is_x_odd, z - y) + y
        is_one = ans == 1
        return ans, is_one

    composition = Wired(
        {
            Wire(Output(collatz, 0), Input(collatz, 0)
        }
    )
```

In this case, the policy states that the first output of the `collatz` function can be forwarded to the first input of `collatz`, but not the second output (which is decrypted every time, and used for control flow).

You can use the `fhe.Wire` between any two functions. It is also possible to define wires  with `fhe.AllInputs` and `fhe.AllOutputs` ends. For instance, in the previous example:
```python
    composition = Wired(
        {
            Wire(AllOutputs(collatz), AllInputs(collatz))
        }
    )
```

This policy would be equivalent to using the `fhe.AllComposable` policy.

## Automatic module tracing

When a module's composition logic is static and straightforward, declaratively defining a `Wired` policy is usually the simplest approach. However, in cases where modules have more complex or dynamic composition logic, deriving an accurate list of `Wire` components to be used in the policy can become challenging.

Another related problem is defining different function input-sets. When the composition logic is simple, these can be provided manually. But as the composition gets more convoluted, computing a consistent ensemble of inputsets for a module may become intractable.

For those advanced cases, you can derive the composition rules and the input-sets automatically from user-provided examples. Consider the following module:
```python
from concrete import fhe
from fhe import Wired

@fhe.module()
class MyModule:
    @fhe.function({"x": "encrypted"})
    def increment(x):
        return (x + 1) % 100

    @fhe.function({"x": "encrypted"})
    def decrement(x):
        return (x - 1) % 100

    @fhe.function({"x": "encrypted"})
    def decimate(x):
        return (x / 10) % 100

    composition = fhe.Wired()
```

You can use the `wire_pipeline` context manager to activate the module tracing functionality:
```python
# A single inputset used during tracing is defined
inputset = [np.random.randint(1, 100, size=()) for _ in range(100)]

# The inputset is passed to the `wire_pipeline` method, which itself returns an iterator over the inputset samples.
with MyModule.wire_pipeline(inputset) as samples_iter:

    # The inputset is iterated over
    for s in samples_iter:

        # Here we provide an example of how we expect the module functions to be used at runtime in fhe.
        MyModule.increment(MyModule.decimate(MyModule.decrement(s)))

# It is not needed to provide any inputsets to the `compile` method after tracing the wires, since those were already computed automatically during the module tracing.
module = MyModule.compile(
    p_error=0.01,
)
```

Note that any dynamic branching is possible during module tracing. However, for complex runtime logic, ensure that the input set provides sufficient examples to cover all potential code paths.

## Current Limitations

Depending on the functions, composition may add a significant overhead compared to a non-composable version.

To be composable, a function must meet the following condition: every output that can be forwarded as input (according to the composition policy) must contain a noise-refreshing operation. Since adding a noise refresh has a noticeable impact on performance, Concrete does not automatically include it.

For instance, to implement a function that doubles an encrypted value, you might write:

```python
@fhe.module()
class Doubler:
    @fhe.compiler({"counter": "encrypted"})
    def double(counter):
       return counter * 2
```
This function is valid with the `fhe.NotComposable` policy. However, if compiled with the `fhe.AllComposable` policy, it will raise a `RuntimeError: Program cannot be composed: ...`, indicating that an extra Programmable Bootstrapping (PBS) step must be added.

To resolve this and make the circuit valid, add a PBS at the end of the circuit:

```python
@fhe.module()
class Doubler:
    @fhe.compiler({"counter": "encrypted"})
    def double(counter):
       return fhe.refresh(counter * 2)
```
