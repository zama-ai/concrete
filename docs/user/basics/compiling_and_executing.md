# Compiling and Executing a Numpy Function

## Importing necessary components

Everything you need to compile and execute homomorphic functions is included in a single module. You can import it like so:

```python
import concrete.numpy as cnp
```

## Defining a function to compile

You need to have a python function that follows the [limits](../explanation/fhe_and_framework_limits.md) of **Concrete Numpy**. Here is a simple example:

<!--pytest-codeblocks:cont-->
```python
def f(x, y):
    return x + y
```

## Compiling the function

To compile the function, you need to identify the inputs that it is expecting. In the example function above, `x` and `y` could be scalars or tensors (though, for now, only dot between tensors are supported), they can be encrypted or clear, they can be signed or unsigned, they can have different bit-widths. So, we need to know what they are beforehand. We can do that like so:

<!--pytest-codeblocks:cont-->
```python
x = "encrypted"
y = "encrypted"
```

In this configuration, both `x` and `y` will be encrypted values.

We also need an inputset. It is to determine the bit-widths of the intermediate results. It should be an iterable yielding tuples in the same order as the inputs of the function to compile. There should be at least 10 inputs in the input set to avoid warnings (except for functions with less than 10 possible inputs). The warning is there because the bigger the input set, the better the bounds will be.

<!--pytest-codeblocks:cont-->
```python
inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]
```

Finally, we can compile our function to its homomorphic equivalent.

<!--pytest-codeblocks:cont-->
```python
compiler = cnp.Compiler(f, {"x": x, "y": y})
circuit = compiler.compile(inputset)

# You can print the compiled circuit:
print(circuit)

# Outputs

# %0 = x                  # EncryptedScalar<uint3>
# %1 = y                  # EncryptedScalar<uint3>
# %2 = add(%0, %1)        # EncryptedScalar<uint4>
# return %2

# Or draw it
circuit.draw(show=True)
```

Here is the graph from the previous code block drawn with `draw`:

![Drawn graph of previous code block](../../_static/howto/compiling_and_executing_example_graph.png)

## Performing homomorphic evaluation

You can use `.encrypt_run_decrypt(...)` method of `Circuit` to perform fully homomorphic evaluation. Here are some examples:

<!--pytest-codeblocks:cont-->
```python
circuit.encrypt_run_decrypt(3, 4)
# 7
circuit.encrypt_run_decrypt(1, 2)
# 3
circuit.encrypt_run_decrypt(7, 7)
# 14
circuit.encrypt_run_decrypt(0, 0)
# 0
```

```{caution}
Be careful about the inputs, though.
If you were to run with values outside the range of the inputset, the result might not be correct.
```

While `.encrypt_run_decrypt(...)` is a good start for prototyping examples, more advanced usages require control over the different steps that are happening behind the scene, mainly key generation, encryption, execution, and decryption. The different steps can of course be called separately as in the example below:

<!--pytest-codeblocks:cont-->
```python
# generate keys required for encrypted computation
circuit.keygen()
# this will encrypt arguments that require encryption and pack all arguments
# as well as public materials (public keys)
public_args = circuit.encrypt(3, 4)
# this will run the encrypted computation using public materials and inputs provided
encrypted_result = circuit.run(public_args)
# the execution returns the encrypted result which can later be decrypted
decrypted_result = circuit.decrypt(encrypted_result)
```

## Further reading

- [Working With Floating Points Tutorial](../tutorial/working_with_floating_points.md)
- [Table Lookup Tutorial](../tutorial/table_lookup.md)
