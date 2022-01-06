# Compiling a Torch Model

**Concrete Numpy** allows you to compile a torch model to its FHE counterpart.


A simple command can compile a torch model to its FHE counterpart. This process executes most of the concepts described in the documentation on [how to use quantization](use_quantization.md) and triggers the compilation to be able to run the model over homomorphically encrypted data.


```python
from torch import nn
import torch
class LogisticRegression(nn.Module):
    """LogisticRegression with Torch"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=14, out_features=1)
        self.sigmoid1 = nn.Sigmoid()


    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.sigmoid1(out)
        return out

torch_model = LogisticRegression()
```

```{warning}
Note that the architecture of the neural network passed to be compiled must respect some hard constraints given by FHE. Please read the our [detailed documentation](../howto/reduce_needed_precision.md) on these limitations.
```

Once your model is trained you can simply call the `compile_torch_model` function to execute the compilation.

<!--pytest-codeblocks:cont-->
```python
from concrete.torch.compile import compile_torch_model
import numpy
torch_input = torch.randn(100, 14)
quantized_numpy_module = compile_torch_model(
    torch_model, # our model
    torch_input, # a representative inputset to be used for both quantization and compilation
    n_bits = 2,
)
```

You can then call `quantized_numpy_module.forward_fhe.run()` to have the FHE inference.

Now your model is ready to infer in FHE settings.

<!--pytest-codeblocks:cont-->
```python
enc_x = numpy.array([numpy.random.randn(14)]).astype(numpy.uint8) # An example that is going to be encrypted, and used for homomorphic inference.
fhe_prediction = quantized_numpy_module.forward_fhe.run(enc_x)
```

`fhe_prediction` contains the clear quantized output. The user can now dequantize the output to get the actual floating point prediction as follows:

<!--pytest-codeblocks:cont-->
```python
clear_output = quantized_numpy_module.dequantize_output(
    numpy.array(fhe_prediction, dtype=numpy.float32)
)
```

If you want to see more compilation examples, you can check out the [Fully Connected Neural Network](../advanced_examples/FullyConnectedNeuralNetwork.ipynb)
