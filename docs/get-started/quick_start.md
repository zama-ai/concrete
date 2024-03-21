# Quick Start

To compute on encrypted data, you first need to define the function you want to compute, then compile it into a Concrete `Circuit`, which you can use to perform homomorphic evaluation.

Here is the full example that we will walk through:

```python
from concrete import fhe

def add(x, y):
    return x + y

compiler = fhe.Compiler(add, {"x": "encrypted", "y": "encrypted"})

inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]

print(f"Compilation...")
circuit = compiler.compile(inputset)

print(f"Key generation...")
circuit.keygen()

print(f"Homomorphic evaluation...")
encrypted_x, encrypted_y = circuit.encrypt(2, 6)
encrypted_result = circuit.run(encrypted_x, encrypted_y)
result = circuit.decrypt(encrypted_result)

assert result == add(2, 6)
```

## Importing the library

Everything you need to perform homomorphic evaluation is included in a single module:

<!--pytest-codeblocks:skip-->
```python
from concrete import fhe
```

## Defining the function to compile

In this example, we compile a simple addition function:

<!--pytest-codeblocks:skip-->
```python
def add(x, y):
    return x + y
```

## Creating a compiler

To compile the function, you need to create a `Compiler` by specifying the function to compile and the encryption status of its inputs:

<!--pytest-codeblocks:skip-->
```python
compiler = fhe.Compiler(add, {"x": "encrypted", "y": "encrypted"})
```

To set that e.g. `y` is in the clear, it would be

<!--pytest-codeblocks:skip-->
```python
compiler = fhe.Compiler(add, {"x": "encrypted", "y": "clear"})
```

## Defining an inputset

An inputset is a collection representing the typical inputs to the function. It is used to determine the bit widths and shapes of the variables within the function.

It should be in iterable, yielding tuples, of the same length as the number of arguments of the function being compiled:

<!--pytest-codeblocks:skip-->
```python
inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]
```

Here, our inputset is made of 10 pairs of integers, whose the minimum pair is `(0, 0)` and the maximum is `(7, 7)`.

{% hint style="warning" %}
Choosing a representative inputset is critical to allow the compiler to find accurate bounds of all the intermediate values (find more details [here](https://docs.zama.ai/concrete/explanations/compilation#bounds-measurement). Later if you evaluate the circuit with values that make under or overflows it results to an undefined behavior.
{% endhint %}

{% hint style="warning" %}
There is a utility function called `fhe.inputset(...)` for easily creating random inputsets, see its
[documentation](../core-features/extensions.md#fheinputset) to learn more!
{% endhint %}

## Compiling the function

You can use the `compile` method of the `Compiler` class with an inputset to perform the compilation and get the resulting circuit back:

<!--pytest-codeblocks:skip-->
```python
print(f"Compilation...")
circuit = compiler.compile(inputset)
```

## Generating the keys

You can use the `keygen` method of the `Circuit` class to generate the keys (public and private):

<!--pytest-codeblocks:skip-->
```python
print(f"Key generation...")
circuit.keygen()
```

{% hint style="info" %}
If you don't call the key generation explicitly keys will be generated lazily when it needed.
{% endhint %}

## Performing homomorphic evaluation

Now you can easily perform the homomorphic evaluation using the `encrypt`, `run` and `decrypt` methods of the `Circuit`:

<!--pytest-codeblocks:skip-->
```python
print(f"Homomorphic evaluation...")
encrypted_x, encrypted_y = circuit.encrypt(2, 6)
encrypted_result = circuit.run(encrypted_x, encrypted_y)
result = circuit.decrypt(encrypted_result)
```
