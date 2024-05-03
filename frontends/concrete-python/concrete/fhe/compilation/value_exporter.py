"""
Declaration of `ValueExporter` class.
"""

# pylint: disable=import-error,no-name-in-module

import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from concrete.compiler import ClientParameters, KeySet, PublicKeySet, SimulatedValueExporter
from concrete.compiler import ValueExporter as _ValueExporter

from ..dtypes import SignedInteger, UnsignedInteger
from ..values import ValueDescription
from .value import Value

# pylint: enable=import-error,no-name-in-module


class ValueExporter:
    """
    ValueExporter is used for encryption python values into Value.
    """

    _exporter: Union[_ValueExporter, SimulatedValueExporter]
    _client_parameters: ClientParameters
    _function_name: str

    def __init__(
        self,
        client_parameters: ClientParameters,
        exporter: Union[_ValueExporter, SimulatedValueExporter],
        function_name: str,
    ):
        if not (isinstance(exporter, (SimulatedValueExporter, _ValueExporter))):
            msg = "value_exporter must be of type SimulatedValueExporter or ValueExporter,"
            f"not {type(exporter)}"
            raise TypeError(msg)
        if not isinstance(client_parameters, ClientParameters):
            msg = "client_parameters must be of type ClientParameters or ValueExporter,"
            f"not {type(client_parameters)}"
            raise TypeError(msg)
        self._exporter = exporter
        self._client_parameters = client_parameters
        self._function_name = function_name

    @staticmethod
    def new_private(
        keyset: KeySet, client_parameters: ClientParameters, function_name: str = "main"
    ):
        """
        Create a new value exporter for private encryption.

        Args:
            function_name (str):
                name of the function to encrypt

        Returns:
            Optional[Union[Value, Tuple[Optional[Value], ...]]]:
                encrypted argument(s) for evaluation
        """
        return ValueExporter(
            client_parameters,
            _ValueExporter.new(keyset, client_parameters, function_name),
            function_name,
        )

    @staticmethod
    def new_public(
        keyset: PublicKeySet, client_parameters: ClientParameters, function_name: str = "main"
    ):
        """
        Create a new value exporter for public encryption.

        Args:
            function_name (str):
                name of the function to encrypt

        Returns:
            Optional[Union[Value, Tuple[Optional[Value], ...]]]:
                encrypted argument(s) for evaluation
        """
        return ValueExporter(
            client_parameters,
            _ValueExporter.new_public(keyset, client_parameters, function_name),
            function_name,
        )

    @staticmethod
    def new_simulated(client_parameters: ClientParameters, function_name: str = "main"):
        """
        Create a new value exporter for simulate encryption.

        Args:
            function_name (str):
                name of the function to encrypt

        Returns:
            Optional[Union[Value, Tuple[Optional[Value], ...]]]:
                encrypted argument(s) for evaluation
        """
        return ValueExporter(
            client_parameters,
            SimulatedValueExporter.new(client_parameters, function_name),
            function_name,
        )

    def encrypt(
        self,
        *args: Optional[Union[int, np.ndarray, List]],
    ) -> Optional[Union[Value, Tuple[Optional[Value], ...]]]:
        """
        Encrypt clear python values to fhe Values.
        """
        print(args)
        ordered_sanitized_args = self._validate_input_args(
            self._client_parameters,
            *args,
        )
        print(ordered_sanitized_args)
        exported = [
            None
            if arg is None
            else Value(
                self._exporter.export_tensor(position, arg.flatten().tolist(), list(arg.shape))
                if isinstance(arg, np.ndarray) and arg.shape != ()
                else self._exporter.export_scalar(position, int(arg))
            )
            for position, arg in enumerate(ordered_sanitized_args)
        ]

        return tuple(exported) if len(exported) != 1 else exported[0]

    def _validate_input_args(
        self,
        client_parameters: ClientParameters,
        *args: Optional[Union[int, np.ndarray, List]],
    ) -> List[Optional[Union[int, np.ndarray]]]:
        """Validate input arguments.

        Args:
            client_specs (ClientSpecs):
                client specification
            *args (Optional[Union[int, np.ndarray, List]]):
                argument(s) for evaluation

        Returns:
            List[Optional[Union[int, np.ndarray]]]: ordered validated args
        """
        functions_parameters = json.loads(client_parameters.serialize())["circuits"]
        client_parameters_json = next(
            filter(lambda x: x["name"] == self._function_name, functions_parameters)
        )
        assert "inputs" in client_parameters_json
        input_specs = client_parameters_json["inputs"]
        if len(args) != len(input_specs):
            message = f"Expected {len(input_specs)} inputs but got {len(args)}"
            raise ValueError(message)

        sanitized_args: Dict[int, Optional[Union[int, np.ndarray]]] = {}
        for index, (arg, spec) in enumerate(zip(args, input_specs)):
            if arg is None:
                sanitized_args[index] = None
                continue

            if isinstance(arg, list):
                arg = np.array(arg)

            is_valid = isinstance(arg, (int, np.integer)) or (
                isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.integer)
            )

            if "lweCiphertext" in spec["typeInfo"].keys():
                type_info = spec["typeInfo"]["lweCiphertext"]
                is_encrypted = True
                shape = tuple(type_info["abstractShape"]["dimensions"])
                assert "integer" in type_info["encoding"].keys()
                width = type_info["encoding"]["integer"]["width"]
                is_signed = type_info["encoding"]["integer"]["isSigned"]
            elif "plaintext" in spec["typeInfo"].keys():
                type_info = spec["typeInfo"]["plaintext"]
                is_encrypted = False
                width = type_info["integerPrecision"]
                is_signed = type_info["isSigned"]
                shape = tuple(type_info["shape"]["dimensions"])
            else:
                message = f"Expected a valid type in {spec['typeInfo'].keys()}"
                raise ValueError(message)

            expected_dtype = SignedInteger(width) if is_signed else UnsignedInteger(width)
            expected_value = ValueDescription(expected_dtype, shape, is_encrypted)
            if is_valid:
                expected_min = expected_dtype.min()
                expected_max = expected_dtype.max()

                if not is_encrypted:
                    # clear integers are signless
                    # (e.g., 8-bit clear integer can be in range -128, 255)
                    expected_min = -(expected_max // 2) - 1

                actual_min = arg if isinstance(arg, int) else arg.min()
                actual_max = arg if isinstance(arg, int) else arg.max()
                actual_shape = () if isinstance(arg, int) else arg.shape

                is_valid = (
                    actual_min >= expected_min
                    and actual_max <= expected_max
                    and actual_shape == expected_value.shape
                )

                if is_valid:
                    sanitized_args[index] = arg

            if not is_valid:
                try:
                    actual_value = str(ValueDescription.of(arg, is_encrypted=is_encrypted))
                except ValueError:
                    actual_value = type(arg).__name__
                message = (
                    f"Expected argument {index} to be {expected_value} but it's {actual_value}"
                )
                raise ValueError(message)

        ordered_sanitized_args = [sanitized_args[i] for i in range(len(sanitized_args))]
        return ordered_sanitized_args
