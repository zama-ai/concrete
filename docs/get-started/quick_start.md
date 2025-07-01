# Quick Start

This document covers how to compute on encrypted data homomorphically using the **Concrete** framework. We will walk you through a complete example step-by-step.



The basic workflow of computation is as follows:
1. Define the function you want to compute
2. Compile the function into a Concrete `Circuit`
3. Use the `Circuit` to perform homomorphic evaluation

Here is the complete example, which we will explain step by step in the following paragraphs.

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

## Decorator

Another simple way to compile a function is to use a decorator.

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x + 42

inputset = range(10)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(10) == f(10)
```

{% hint style="info" %}
This decorator is a way to add the `compile` method to the function object without changing its name elsewhere.
{% endhint %}

## Importing the library

Import the `fhe` module, which includes everything you need to perform homomorphic evaluation:

<!--pytest-codeblocks:skip-->
```python
from concrete import fhe
```

## Defining the function to compile

Here we define a simple addition function:

<!--pytest-codeblocks:skip-->
```python
def add(x, y):
    return x + y
```

## Creating a compiler

To compile the function, you first need to create a `Compiler` by specifying the function to compile and the encryption status of its inputs:

<!--pytest-codeblocks:skip-->
```python
compiler = fhe.Compiler(add, {"x": "encrypted", "y": "encrypted"})
```

For instance, to set the input y as clear:

<!--pytest-codeblocks:skip-->
```python
compiler = fhe.Compiler(add, {"x": "encrypted", "y": "clear"})
```

## Defining an inputset

An inputset is a collection representing the typical inputs of the function. It is used to determine the bit widths and shapes of the variables within the function.

The inputset should be an iterable that yields tuples of the same length as the number of arguments of the compiled function. 

For example:

<!--pytest-codeblocks:skip-->
```python
inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]
```

Here, our inputset consists of 10 integer pairs, ranging from a minimum of `(0, 0)` to a maximum of `(7, 7)`.

{% hint style="warning" %}
Choosing a representative inputset is critical to allow the compiler to find accurate bounds of all the intermediate values (see more details [here](https://docs.zama.ai/concrete/explanations/compiler_workflow#bounds-measurement)). Evaluating the circuit with input values under or over the bounds may result in undefined behavior.
{% endhint %}

{% hint style="warning" %}
You can use the `fhe.inputset(...)` function to easily create random inputsets, see more details in [this documentation](../core-features/extensions.md#fheinputset).
{% endhint %}

## Compiling the function

Use the `compile` method of the `Compiler` class with an inputset to perform the compilation and get the resulting circuit:

<!--pytest-codeblocks:skip-->
```python
print(f"Compilation...")
circuit = compiler.compile(inputset)
```

## Generating the keys

Use the `keygen` method of the `Circuit` class to generate the keys (public and private):

<!--pytest-codeblocks:skip-->
```python
print(f"Key generation...")
circuit.keygen()
```

{% hint style="info" %}
If you don't call the key generation explicitly, keys will be generated lazily when needed.
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
