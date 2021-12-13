"""QuantizedModule API."""
import copy
from typing import Optional, Union

import numpy

from ..common.compilation.artifacts import CompilationArtifacts
from ..common.compilation.configuration import CompilationConfiguration
from ..common.fhe_circuit import FHECircuit
from ..numpy.np_fhe_compiler import NPFHECompiler
from .quantized_array import QuantizedArray


class QuantizedModule:
    """Inference for a quantized model."""

    quant_layers_dict: dict
    _mode: str
    q_input: Optional[QuantizedArray]
    forward_fhe: Union[None, FHECircuit]

    def __init__(self, quant_layers_dict: dict):
        self.quant_layers_dict = copy.deepcopy(quant_layers_dict)
        self.compiled = False
        self.forward_fhe = None
        self.q_input = None

    def __call__(self, x: QuantizedArray):
        return self.forward(x)

    def forward(self, q_x: Union[numpy.ndarray, QuantizedArray]) -> numpy.ndarray:
        """Forward pass with numpy function only.

        Args:
            q_x (Union[numpy.ndarray, QuantizedArray]): QuantizedArray containing the inputs
                                                        or a numpy.array containing the q_values.
                                                        In the latter, the stored input parameters
                                                        are used:
                                                        (q_input.scale, q_input.zero_point).

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        # Following "if not" important for compilation as the tracer
        # need to fall in it the statement (tracing).
        # If the q_x is a numpy module then we reuse self.q_input parameters
        # computed during calibration.
        # Later we might want to only allow nympy.array input
        if not isinstance(q_x, QuantizedArray):
            assert self.q_input is not None
            self.q_input.update_qvalues(q_x)
            q_x = self.q_input

        for _, layer in self.quant_layers_dict.items():
            q_x = layer(q_x)

        # mypy compliance
        assert isinstance(q_x, QuantizedArray)

        return q_x.qvalues

    def forward_and_dequant(self, q_x: Union[numpy.ndarray, QuantizedArray]) -> numpy.ndarray:
        """Forward pass with numpy function only plus dequantization.

        Args:
            q_x (Union[numpy.ndarray, QuantizedArray]): QuantizedArray containing the inputs
                                                        or a numpy.array containing the q_values.
                                                        In the latter, the stored input parameters
                                                        are used:
                                                        (q_input.scale, q_input.zero_point).

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        q_out = self.forward(q_x)
        return self.dequantize_output(q_out)

    def dequantize_output(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Take the last layer q_out and use its dequant function.

        Args:
            qvalues (numpy.ndarray): Quantized values of the last layer.

        Returns:
            numpy.ndarray: Dequantized values of the last layer.
        """
        last_layer = list(self.quant_layers_dict.values())[-1]
        real_values = last_layer.q_out.update_qvalues(qvalues)
        return real_values

    def compile(
        self,
        q_input: QuantizedArray,
        compilation_configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
    ) -> FHECircuit:
        """Compile the forward function of the module.

        Args:
            q_input (QuantizedArray): Needed for tracing and building the boundaries.
            compilation_configuration (Optional[CompilationConfiguration]): Configuration object
                                                                            to use during
                                                                            compilation
            compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill during
                                                                    compilation
            show_mlir (bool, optional): if set, the MLIR produced by the converter and which is
                going to be sent to the compiler backend is shown on the screen, e.g., for debugging
                or demo. Defaults to False.

        Returns:
            FHECircuit: the compiled FHECircuit.
        """

        self.q_input = copy.deepcopy(q_input)
        compiler = NPFHECompiler(
            self.forward,
            {
                "q_x": "encrypted",
            },
            compilation_configuration,
            compilation_artifacts,
        )
        self.forward_fhe = compiler.compile_on_inputset(
            (numpy.expand_dims(arr, 0) for arr in self.q_input.qvalues), show_mlir
        )

        return self.forward_fhe
