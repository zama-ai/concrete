# Having a Function Which Requires Less Precision

With our current technology, we cannot represent integers with more than 7 bits.
We are actively working on supporting larger integers, so it should get better in the future.

## What happens when you have larger values?

You get a compilation error. Here is an example:

<!--python-test:skip-->
```python
import concrete.numpy as hnp

def f(x):
    return 42 * x

compiler = hnp.NPFHECompiler(f, {"x": "encrypted"})
compiler.eval_on_inputset(range(2 ** 3))
compiler.get_compiled_fhe_circuit()
```

results in

```
Traceback (most recent call last):
  File "/home/default/Documents/Projects/Zama/hdk/dist/demo.py", line 9, in <module>
    circuit = compiler.get_compiled_fhe_circuit()
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/np_fhe_compiler.py", line 274, in get_compiled_fhe_circuit
    return compile_op_graph_to_fhe_circuit(
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/compile.py", line 676, in compile_op_graph_to_fhe_circuit
    result = run_compilation_function_with_error_management(
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/compile.py", line 141, in run_compilation_function_with_error_management
    return compilation_function()
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/compile.py", line 674, in compilation_function
    return _compile_op_graph_to_fhe_circuit_internal(op_graph, show_mlir, compilation_artifacts)
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/compile.py", line 626, in _compile_op_graph_to_fhe_circuit_internal
    prepare_op_graph_for_mlir(op_graph)
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/compile.py", line 603, in prepare_op_graph_for_mlir
    update_bit_width_for_mlir(op_graph)
  File "/home/default/Documents/Projects/Zama/hdk/concrete/common/mlir/utils.py", line 204, in update_bit_width_for_mlir
    raise RuntimeError(
RuntimeError: max_bit_width of some nodes is too high for the current version of the compiler (maximum must be 7) which is not compatible with:

%0 = x                  # EncryptedScalar<uint3>
%1 = 42                 # ClearScalar<uint6>
%2 = mul(%0, %1)        # EncryptedScalar<uint9>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 9 bits is not supported for the time being
return %2
```

when you try to run.

## Why can some computation work with less precision?

### The input data uses more bits than required

For some tasks, like classification for example, the output prediction often carries much less information than the input data used to make that prediction.

For example the MNIST classification task consists in taking an image, a 28x28 array containing uint8 values, representing a number and predict whether it belongs to one of 10 classes: the numbers from 0 to 9 included. The output is a one-hot vector to indicate which class a particular sample belongs to.

The input contains 28x28x8 = 6272 bits of information. In practice you could also get good results on MNIST by binarizing the images and training a model for that Binarized MNIST task. This means that in a real use case where you actually need to do digits recognition, you could binarize your input on the fly, replacing each pixel by either 0 or 1. Doing so, you use 1 bit per pixel and now only have 768 bits of input data. It also means that if you are doing some accumulation (adding pixel values together), you are going to need accumulators that are smaller (adding 0s and 1s requires less space than adding values ranging from 0 to 255 included).

This shows how adapting your data can allow you to use models that may require smaller data types (i.e. use less precision) to perform their computations.

```{note}
Binarizing here is akin to quantization which is introduced [here](../explanation/QUANTIZATION.md). You can also find further resources on the linked page.
```

### There is a tolerance on the result

If for some reason you have a tolerance on the result's precision, then you can change the computation used to a certain extent and still be in that tolerance range.

This is illustrated in both advanced examples [Quantized Linear Regression](../advanced_examples/QuantizedLinearRegression.ipynb) and [Quantized Logistic Regression](../advanced_examples/QuantizedLogisticRegression.ipynb).

The end result has a granularity/imprecision linked to the data types used and for the Quantized Logistic Regression to the lattice used to evaluate the logistic model.

```{warning}
FIXME(jordan/andrei): update with your insights / knowledge on ML
```
