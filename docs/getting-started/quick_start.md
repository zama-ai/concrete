# Quick Start

To compute on encrypted data, you first need to define the function you want to compute, then compile it into a Concrete `Circuit`, which you can use to perform homomorphic evaluation.

Here is the full example that we will walk through:

```python
from concrete import fhe

def add(x, y):
    return x + y

compiler = fhe.Compiler(add, {"x": "encrypted", "y": "clear"})

inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1)]
circuit = compiler.compile(inputset)

x = 4
y = 4

clear_evaluation = add(x, y)
homomorphic_evaluation = circuit.encrypt_run_decrypt(x, y)

print(x, "+", y, "=", clear_evaluation, "=", homomorphic_evaluation)
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
compiler = fhe.Compiler(add, {"x": "encrypted", "y": "clear"})
```

## Defining an inputset

An inputset is a collection representing the typical inputs to the function. It is used to determine the bit widths and shapes of the variables within the function.

It should be in iterable, yielding tuples, of the same length as the number of arguments of the function being compiled:

<!--pytest-codeblocks:skip-->
```python
inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1)]
```

{% hint style="warning" %}
All inputs in the inputset will be evaluated in the graph, which takes time. If you're experiencing long compilation times, consider providing a smaller inputset.
{% endhint %}

{% hint style="warning" %}
There is a utility function called `fhe.inputset(...)` for easily creating random inputsets, see its
[documentation](../tutorial/extensions.md#fheinputset) to learn more!
{% endhint %}

## Compiling the function

You can use the `compile` method of a `Compiler` class with an inputset to perform the compilation and get the resulting circuit back:

<!--pytest-codeblocks:skip-->
```python
circuit = compiler.compile(inputset)
```

## Performing homomorphic evaluation

You can use the `encrypt_run_decrypt` method of a `Circuit` class to perform homomorphic evaluation:

<!--pytest-codeblocks:skip-->
```python
homomorphic_evaluation = circuit.encrypt_run_decrypt(4, 4)
```

{% hint style="info" %}
`circuit.encrypt_run_decrypt(*args)` is just a convenient way to do everything at once. It is implemented as `circuit.decrypt(circuit.run(circuit.encrypt(*args)))`.
{% endhint %}
