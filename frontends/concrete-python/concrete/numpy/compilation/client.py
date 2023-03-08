"""
Declaration of `Client` class.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from concrete.compiler import (
    ClientSupport,
    EvaluationKeys,
    KeySet,
    KeySetCache,
    PublicArguments,
    PublicResult,
)

from ..dtypes.integer import SignedInteger, UnsignedInteger
from ..internal.utils import assert_that
from ..values.value import Value
from .specs import ClientSpecs


class Client:
    """
    Client class, which can be used to manage keys, encrypt arguments and decrypt results.
    """

    specs: ClientSpecs

    _keyset: Optional[KeySet]
    _keyset_cache: Optional[KeySetCache]

    def __init__(
        self,
        client_specs: ClientSpecs,
        keyset_cache_directory: Optional[Union[str, Path]] = None,
    ):
        self.specs = client_specs

        self._keyset = None
        self._keyset_cache = None

        if keyset_cache_directory is not None:
            self._keyset_cache = KeySetCache.new(str(keyset_cache_directory))

    def save(self, path: Union[str, Path]):
        """
        Save the client into the given path in zip format.

        Args:
            path (Union[str, Path]):
                path to save the client
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(Path(tmp_dir) / "client.specs.json", "w", encoding="utf-8") as f:
                f.write(self.specs.serialize())

            path = str(path)
            if path.endswith(".zip"):
                path = path[: len(path) - 4]

            shutil.make_archive(path, "zip", tmp_dir)

    @staticmethod
    def load(
        path: Union[str, Path],
        keyset_cache_directory: Optional[Union[str, Path]] = None,
    ) -> "Client":
        """
        Load the client from the given path in zip format.

        Args:
            path (Union[str, Path]):
                path to load the client from

            keyset_cache_directory (Optional[Union[str, Path]], default = None):
                keyset cache directory to use

        Returns:
            Client:
                client loaded from the filesystem
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.unpack_archive(path, tmp_dir, "zip")
            with open(Path(tmp_dir) / "client.specs.json", "r", encoding="utf-8") as f:
                client_specs = ClientSpecs.unserialize(f.read())

        return Client(client_specs, keyset_cache_directory)

    def keygen(self, force: bool = False):
        """
        Generate keys required for homomorphic evaluation.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated
        """

        if self._keyset is None or force:
            self._keyset = ClientSupport.key_set(self.specs.client_parameters, self._keyset_cache)

    def encrypt(self, *args: Union[int, np.ndarray]) -> PublicArguments:
        """
        Prepare inputs to be run on the circuit.

        Args:
            *args (Union[int, numpy.ndarray]):
                inputs to the circuit

        Returns:
            PublicArguments:
                encrypted and plain arguments as well as public keys
        """

        client_parameters_json = json.loads(self.specs.client_parameters.serialize())
        assert_that("inputs" in client_parameters_json)
        input_specs = client_parameters_json["inputs"]

        if len(args) != len(input_specs):
            message = f"Expected {len(input_specs)} inputs but got {len(args)}"
            raise ValueError(message)

        sanitized_args: Dict[int, Union[int, np.ndarray]] = {}
        for index, spec in enumerate(input_specs):
            arg = args[index]
            if isinstance(arg, list):
                arg = np.array(arg)

            is_valid = isinstance(arg, (int, np.integer)) or (
                isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.integer)
            )

            width = spec["shape"]["width"]
            shape = tuple(spec["shape"]["dimensions"])
            is_encrypted = spec["encryption"] is not None

            expected_dtype = (
                SignedInteger(width) if self.specs.input_signs[index] else UnsignedInteger(width)
            )
            expected_value = Value(expected_dtype, shape, is_encrypted)
            if is_valid:
                expected_min = expected_dtype.min()
                expected_max = expected_dtype.max()

                actual_min = arg if isinstance(arg, int) else arg.min()
                actual_max = arg if isinstance(arg, int) else arg.max()
                actual_shape = () if isinstance(arg, int) else arg.shape

                is_valid = (
                    actual_min >= expected_min
                    and actual_max <= expected_max
                    and actual_shape == expected_value.shape
                )

                if is_valid:
                    is_signed = self.specs.input_signs[index]
                    sanitizer = 0 if not is_signed else 2 ** (width - 1)

                    if isinstance(arg, int):
                        sanitized_args[index] = arg + sanitizer
                    else:
                        sanitized_args[index] = (arg + sanitizer).astype(np.uint64)

            if not is_valid:
                actual_value = Value.of(arg, is_encrypted=is_encrypted)
                message = (
                    f"Expected argument {index} to be {expected_value} but it's {actual_value}"
                )
                raise ValueError(message)

        self.keygen(force=False)
        return ClientSupport.encrypt_arguments(
            self.specs.client_parameters,
            self._keyset,
            [sanitized_args[i] for i in range(len(sanitized_args))],
        )

    def decrypt(
        self,
        result: PublicResult,
    ) -> Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
        """
        Decrypt result of homomorphic evaluaton.

        Args:
            result (PublicResult):
                encrypted result of homomorphic evaluaton

        Returns:
            Union[int, numpy.ndarray]:
                clear result of homomorphic evaluaton
        """

        self.keygen(force=False)
        outputs = ClientSupport.decrypt_result(self.specs.client_parameters, self._keyset, result)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        sanitized_outputs: List[Union[int, np.ndarray]] = []

        client_parameters_json = json.loads(self.specs.client_parameters.serialize())
        assert_that("outputs" in client_parameters_json)
        output_specs = client_parameters_json["outputs"]

        for index, output in enumerate(outputs):
            is_signed = self.specs.output_signs[index]
            crt_decomposition = (
                output_specs[index].get("encryption", {}).get("encoding", {}).get("crt", [])
            )

            if is_signed:
                if crt_decomposition:
                    if isinstance(output, int):
                        sanititzed_output = (
                            output
                            if output < (int(np.prod(crt_decomposition)) // 2)
                            else -int(np.prod(crt_decomposition)) + output
                        )
                    else:
                        output = output.astype(np.longlong)  # to prevent overflows in numpy
                        sanititzed_output = np.where(
                            output < (np.prod(crt_decomposition) // 2),
                            output,
                            -np.prod(crt_decomposition) + output,
                        ).astype(
                            np.int64
                        )  # type: ignore

                    sanitized_outputs.append(sanititzed_output)

                else:
                    n = output_specs[index]["shape"]["width"]
                    output %= 2**n
                    if isinstance(output, int):
                        sanititzed_output = output if output < (2 ** (n - 1)) else output - (2**n)
                        sanitized_outputs.append(sanititzed_output)
                    else:
                        output = output.astype(np.longlong)  # to prevent overflows in numpy
                        sanititzed_output = np.where(
                            output < (2 ** (n - 1)), output, output - (2**n)
                        ).astype(
                            np.int64
                        )  # type: ignore
                        sanitized_outputs.append(sanititzed_output)
            else:
                sanitized_outputs.append(
                    output if isinstance(output, int) else output.astype(np.uint64)
                )

        return sanitized_outputs[0] if len(sanitized_outputs) == 1 else tuple(sanitized_outputs)

    @property
    def evaluation_keys(self) -> EvaluationKeys:
        """
        Get evaluation keys for encrypted computation.

        Returns:
            EvaluationKeys
                evaluation keys for encrypted computation
        """

        self.keygen(force=False)

        assert self._keyset is not None
        return self._keyset.get_evaluation_keys()
