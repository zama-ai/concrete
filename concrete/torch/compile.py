"""torch compilation function."""

from typing import Optional

import numpy
import torch

from ..common.compilation import CompilationArtifacts, CompilationConfiguration
from ..quantization import PostTrainingAffineQuantization, QuantizedArray
from . import NumpyModule


def compile_torch_model(
    torch_model: torch.nn.Module,
    torch_inputset: torch.FloatTensor,
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
    show_mlir: bool = False,
    n_bits=7,
):
    """Take a model in torch, turn it to numpy, transform weights to integer.

    Later, we'll compile the integer model.

    Args:
        torch_model (torch.nn.Module): the model to quantize,
        torch_inputset (torch.FloatTensor): the inputset, in torch form
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization

    """

    # Create corresponding numpy model
    numpy_model = NumpyModule(torch_model)

    # Torch input to numpy
    numpy_inputset = numpy.array(
        [
            tuple(val.cpu().numpy() for val in input_)
            if isinstance(input_, tuple)
            else tuple(input_.cpu().numpy())
            for input_ in torch_inputset
        ]
    )

    # Quantize with post-training static method, to have a model with integer weights
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_model)
    quantized_model = post_training_quant.quantize_module(numpy_inputset)
    model_to_compile = quantized_model

    # Quantize input
    quantized_numpy_inputset = QuantizedArray(n_bits, numpy_inputset)

    # FIXME: just print, to avoid to have useless vars. Will be removed once we can finally compile
    # the model
    print(
        model_to_compile,
        quantized_numpy_inputset,
        compilation_configuration,
        compilation_artifacts,
        show_mlir,
    )
