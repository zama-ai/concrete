# Compiling and Executing

## Importing necessary components

Everything you need to compile and execute homomorphic functions is included in a single module. You can import it like so:

```python
import concrete.numpy as hnp
```

## Defining a function to compile

You need to have a python function that follows the [limits](../explanation/FHE_AND_FRAMEWORK_LIMITS.md) of the Concrete Framework. Here is a simple example:

```python
def f(x, y):
    return x + y
```

## Compiling the function

To compile the function, you need to provide what are the inputs that it's expecting. In the example function above, `x` and `y` could be scalars or tensors (though, for now, only dot between tensors are supported), they can be encrypted or clear, they can be signed or unsigned, they can have different bit-widths. So, we need to know what they are beforehand. We can do that like so:

```python
x = hnp.EncryptedScalar(hnp.UnsignedInteger(3))
y = hnp.EncryptedScalar(hnp.UnsignedInteger(3))
```

In this configuration, both `x` and `y` are 3-bit unsigned integers, so they have the range of `[0, 2**3 - 1]`

We also need a dataset. However, it's not the dataset used in traning as it doesn't contain any labels. It is to determine the bit-widths of the intermediate results so only the inputs are necessary. It should be an iterable yielding tuples in the same order as the inputs of the function to compile.

```python
dataset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1)]
```

Finally, we can compile our function to its homomorphic equivalent.

```python
engine = hnp.compile_numpy_function(
    f, {"x": x, "y": y},
    dataset=iter(dataset),
)
```

## Performing homomorphic evaluation

You can use `.run(...)` method of `engine` returned by `hnp.compile_numpy_function(...)` to perform fully homomorphic evaluation. Here are some examples:

```python
>>> engine.run(3, 4)
7
>>> engine.run(1, 2)
3
>>> engine.run(7, 7)
14
>>> engine.run(0, 0)
0
```

Be careful about the inputs, though.
If you were to run with values outside the range of the dataset, the result might not be correct.

## Further reading

- [Arithmetic Operations Tutorial](../tutorial/ARITHMETIC_OPERATIONS.md)
- [Working With Floating Points Tutorial](../tutorial/WORKING_WITH_FLOATING_POINTS.md)
- [Table Lookup Tutorial](../tutorial/TABLE_LOOKUP.md)
