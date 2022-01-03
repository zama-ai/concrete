"""torch compilation function."""

from typing import Iterable, Optional, Union

import numpy
import torch

from ..common.compilation import CompilationArtifacts, CompilationConfiguration
from ..quantization import PostTrainingAffineQuantization, QuantizedArray, QuantizedModule
from . import NumpyModule

TorchDataset = Iterable[torch.Tensor]
NPDataset = Iterable[numpy.ndarray]


def convert_torch_tensor_or_numpy_array_to_numpy_array(
    torch_tensor_or_numpy_array: Union[torch.Tensor, numpy.ndarray]
) -> numpy.ndarray:
    """Convert a torch tensor or a numpy array to a numpy array.

    Args:
        torch_tensor_or_numpy_array (Union[torch.Tensor, numpy.ndarray]): the value that is either
            a torch tensor or a numpy array.

    Returns:
        numpy.ndarray: the value converted to a numpy array.
    """
    return (
        torch_tensor_or_numpy_array
        if isinstance(torch_tensor_or_numpy_array, numpy.ndarray)
        else torch_tensor_or_numpy_array.cpu().numpy()
    )


def compile_torch_model(
    torch_model: torch.nn.Module,
    torch_inputset: Union[TorchDataset, NPDataset],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
    show_mlir: bool = False,
    n_bits=7,
) -> QuantizedModule:
    """Take a model in torch, turn it to numpy, transform weights to integer.

    Later, we'll compile the integer model.

    Args:
        torch_model (torch.nn.Module): the model to quantize,
        torch_inputset (Union[TorchDataset, NPDataset]): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        function_parameters_encrypted_status (Dict[str, Union[str, EncryptedStatus]]): a dict with
            the name of the parameter and its encrypted status
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """

    # Create corresponding numpy model
    numpy_model = NumpyModule(torch_model)

    # Torch input to numpy
    numpy_inputset_as_single_array = numpy.concatenate(
        tuple(
            numpy.expand_dims(convert_torch_tensor_or_numpy_array_to_numpy_array(input_), 0)
            for input_ in torch_inputset
        )
    )

    # Quantize with post-training static method, to have a model with integer weights
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_model, is_signed=True)
    quantized_module = post_training_quant.quantize_module(numpy_inputset_as_single_array)

    # Quantize input
    quantized_numpy_inputset = QuantizedArray(n_bits, numpy_inputset_as_single_array)

    quantized_module.compile(
        quantized_numpy_inputset,
        compilation_configuration,
        compilation_artifacts,
        show_mlir,
    )

    return quantized_module
