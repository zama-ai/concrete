# Composing functions with modules

In various cases, deploying a server that contains many compatible functions is important. `concrete-python` is now able to compile FHE _modules_, which can contain as many functions as needed. More importantly, modules support _composition_ of the different functions. This means the encrypted result of one function execution can be used as input of a different function, without needing to decrypt in between. A module is [deployed in a single artifact](../guides/deploy.md#deployment-of-modules), making as simple to use a single function project.

Here is a first simple example:
```python
from concrete import fhe

@fhe.module()
class Counter:
    @fhe.function({"x": "encrypted"})
    def inc(x):
        return x + 1 % 20

    @fhe.function({"x": "encrypted"})
    def dec(x):
        return x - 1 % 20
```

You can compile the FHE module `Counter` using the `compile` method. To do that, you need to provide a dictionnary of input sets for every function:

```python
inputset = list(range(20))
CounterFhe = CounterFhe.compile({"inc": inputset, "dec": inputset})
```

After the module has been compiled, we can encrypt and call the different functions in the following way:

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

## Multi inputs, multi outputs

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

## Iteration support

With the previous example we see that to some extent, modules allows to support iteration with cleartext iterands. That is, loops with the following shape :

```python
for i in some_cleartext_constant_range:
    # Do something in FHE in the loop body, implemented as an FHE function.
```

With this pattern, we can also support unbounded loops or complex dynamic condition, as long as this condition is computed in pure cleartext python. Here is an example that computes the [Collatz sequence](https://en.wikipedia.org/wiki/Collatz_conjecture):

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
CollatzFhe = collatz.compile({"collatz": inputset})

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

Which prints:

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

Here we use a while loop that keeps iterating as long as the decryption of the running value is different from `1`. Again, the loop body is implemented in FHE, but the iteration control has to be in the clear.

## Optimizing runtimes with composition policies

By default when using modules, every inputs and outputs of every functions are compatible: they share the same precision and the same crypto-parameters. This means that the most costly crypto-parameters of all code-paths is used for every code paths. This simplicity comes at a cost, and depending on the use case, it may not be necessary.

To optimize the runtimes, we provide a finer grained control over the composition policy via the `composition` module attribute. Here is an example:
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

By default the attribute is set to `fhe.AllComposable`. This policy ensures that every ciphertexts used in the module are compatible. This is the less restrictive, but most costly policy.

If one does not need composition at all, but just want to pack multiple functions in a single artifact, it is possible to do so by setting the `composition` attribute to `fhe.NotComposable`. This is the most restrictive, but less costly policy.

Hopefully there is no need to choose between one of those two extremes. It is also possible to detail custom policies by using `fhe.Wired`. For instance:
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
        [
            Wire(Output(collatz, 0), Input(collatz, 0)
        ]
    )
```

In this case, the policy states that the first output of the `collatz` function can be forwarded to the first input of `collatz`, but not the second output (which is decrypted every time, and used for control flow).

It is possible to use an `fhe.Wire` between any two functions, it is also possible to define wires  with `fhe.AllInputs` and `fhe.AllOutputs` ends. For instance in the previous example:
```python
    composition = Wired(
        [
            Wire(AllOutputs(collatz), AllInputs(collatz))
        ]
    )
```

This policy would be equivalent to using the `fhe.AllComposable` policy.

## Limitations

Depending on the functions, supporting composition may add a non-negligible overhead when compared to a non-composable version. Indeed, to be composable a function must verify the following condition: Every output which can be forwarded as input (as per the composition policy) must contain a noise refreshing operation.

Since adding a noise refresh has a non negligeable impact on performance, `concrete-python` does not do it in behalf of the user. For instance, to implement a function that doubles an encrypted value, we would write something like:

```python
@fhe.module()
class Doubler:
    @fhe.compiler({"counter": "encrypted"})
    def double(counter):
       return counter * 2
```

This is a valid function with the `fhe.NotComposable` policy, but if compiled with `fhe.AllComposable` policy, a `RuntimeError: Program can not be composed: ...` error is reported, signalling that an extra PBS must be added. To solve this situation, and turn this circuit into a valid one, one can use the following snippet to add a PBS at the end of the circuit:

```python
def noise_reset(x):
   return fhe.univariate(lambda x: x)(x)

@fhe.module()
class Doubler:
    @fhe.compiler({"counter": "encrypted"})
    def double(counter):
       return noise_reset(counter * 2)
```

## Single function composition without modules.

It is also possible to compile a single function to be self-composable with the `fhe.AllComposable` policy without using modules. For this one simply has to set the [`composable`](../guides/configure.md#options) configuration setting to `True` when compiling.
