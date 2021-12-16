# Computation With Quantized Functions

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
circuit = compiler.compile_on_inputset(range(2 ** 3))
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

For example the MNIST classification task consists in taking an image, a 28x28 array containing uint8 values, representing a handwritten digit and predicting whether it belongs to one of 10 classes: the digits from 0 to 9. The output is a one-hot vector which indicates the class a particular sample belongs to.

The input contains 28x28x8 = 6272 bits of information. In practice you could still obtain good results on MNIST by thresholding the pixels to {0, 1} and training a model for this new Binarized MNIST task. This means that in a real use case where you actually need to do digits recognition, you could binarize your input on the fly, replacing each pixel by either 0 or 1. Doing so, you use 1 bit per pixel and now only have 784 bits of input data. It also means that if you are doing some accumulation (adding pixel values together), you are going to need accumulators that are smaller (adding 0s and 1s requires less space than adding values ranging from 0 to 255 included).

This shows how adapting your data can allow you to use models that may require smaller data types (i.e. use less precision) to perform their computations.

```{note}
Binarizing here is an extreme case of quantization which is introduced [here](../explanation/quantization.md). You can also find further resources on the linked page.
```

### Model accuracy when quantizing for FHE

Quantization and binarization increase inference speed, reduce model byte-size and are required to run computation in FHE. However, quantization and, especially, binarization, induce a loss in the accuracy of the model since it's representation power is diminished. Choosing quantization parameters carefully can alleviate the accuracy loss all the while allowing compilation to FHE.

This is illustrated in both advanced examples [Quantized Linear Regression](../advanced_examples/QuantizedLinearRegression.ipynb) and [Quantized Logistic Regression](../advanced_examples/QuantizedLogisticRegression.ipynb).

The end result has a granularity/imprecision linked to the data types used and for the Quantized Logistic Regression to the lattice used to evaluate the logistic model.

## Limitations for FHE friendly neural network 

Recent quantization literature often takes a few shortcuts to reach performance similar to those achieved by floating point models. A common one is that the input is left in floating point. This is also true for the first and last layers which have more impact on the resulting model accuracy than hidden layers. 

But, in the Concrete framework the inputs, weights and the accumulator must remain on a maximum of 7 bits.

Thus, in the Concrete framework we also quantize the input data and network output activations in the same way as the rest of the network: everything is quantized to a specific number of bits. It turns out, that the number of bits used for the input or the output of any activation function is crucial to comply with the constraint on accumulator width.

The core operation in neural networks is essentially matrix multiplications (matmul). This operation must be done such that the maximum value of its result requires at most 7 bits of precision.

For example, if you quantize your input and weights with $ n_{\mathsf{weights}} $, $ n_{\mathsf{inputs}} $  bits of precision, one can compute the maximum dimensionality of the input and weights before the matmul **can** exceed the 7 bits as such:

$$ \Omega = \mathsf{floor} \left( \frac{2^{n_{\mathsf{max}}} - 1}{(2^{n_{\mathsf{weights}}} - 1)(2^{n_{\mathsf{inputs}}} - 1)} \right) $$

where $ n_{\mathsf{max}} = 7 $ is the maximum precision allowed. For example, if we set $ n_{\mathsf{weights}} = 2$ and $ n_{\mathsf{inputs}} = 2$ with $ n_{\mathsf{max}} = 7$ then we have the $ \Omega = 14 $ different inputs/weights allowed in the matmul. 

Above $ \Omega $ dimensions in the input and weights, the risk of overflow increases quickly. It may happen that for some distributions of weights and values the computation does not overflow, but the risk increases rapidly with the number of dimensions.

Currently, Concrete Framework pre-computes the number of bits needed for the computation depending on the input set calibration data and does not allow the overflow[^1] to happen.

[^1]: [Integer overflow](https://en.wikipedia.org/wiki/Integer_overflow) 