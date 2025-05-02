"""
Declaration of `Client` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from concrete.compiler import ClientProgram, LweSecretKey
from concrete.compiler import Value as Value_

from .evaluation_keys import EvaluationKeys
from .keys import Keys
from .specs import ClientSpecs
from .utils import validate_input_args
from .value import Value

# pylint: enable=import-error,no-member,no-name-in-module


class Client:
    """
    Client class, which can be used to manage keys, encrypt arguments and decrypt results.
    """

    _client_specs: ClientSpecs
    _keys: Optional[Keys]

    def __init__(
        self,
        client_specs: ClientSpecs,
        keyset_cache_directory: Optional[Union[str, Path]] = None,
        is_simulated: bool = False,
    ):
        self._client_specs = client_specs
        self._keys = None
        if not is_simulated:
            self._keys = Keys(client_specs, keyset_cache_directory)

    def save(self, path: Union[str, Path]):
        """
        Save the client into the given path in zip format.

        Args:
            path (Union[str, Path]):
                path to save the client
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(Path(tmp_dir) / "client.specs.json", "wb") as f:
                f.write(self._client_specs.serialize())

            path = str(path)
            if path.endswith(".zip"):
                path = path[: len(path) - 4]

            shutil.make_archive(path, "zip", tmp_dir)

    @staticmethod
    def load(
        path: Union[str, Path],
        keyset_cache_directory: Optional[Union[str, Path]] = None,
        is_simulated: bool = False,
    ) -> "Client":
        """
        Load the client from the given path in zip format.

        Args:
            path (Union[str, Path]):
                path to load the client from

            keyset_cache_directory (Optional[Union[str, Path]], default = None):
                keyset cache directory to use

            is_simulated (bool, default = False):
                should perform

        Returns:
            Client:
                client loaded from the filesystem
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.unpack_archive(path, tmp_dir, "zip")
            with open(Path(tmp_dir) / "client.specs.json", "rb") as f:
                client_specs = ClientSpecs.deserialize(f.read())

        return Client(client_specs, keyset_cache_directory, is_simulated)

    @property
    def specs(self) -> ClientSpecs:
        """
        Get the client specs for the client.
        """
        return self._client_specs

    @specs.setter
    def specs(self, new_spec: ClientSpecs):
        """
        Get the spec for the client.
        """
        self._client_specs = new_spec

    @property
    def keys(self) -> Optional[Keys]:
        """
        Get the keys for the client.
        """
        return self._keys

    @keys.setter
    def keys(self, new_keys: Keys):
        """
        Set the keys for the client.
        """
        assert self._keys is not None, "Tried to set keys on simulated client."
        assert new_keys.are_generated, "Keyset is not generated."
        self._keys = new_keys

    def keygen(
        self,
        force: bool = False,
        secret_seed: Optional[int] = None,
        encryption_seed: Optional[int] = None,
        initial_keys: Optional[dict[int, LweSecretKey]] = None,
    ):
        """
        Generate keys required for homomorphic evaluation.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated

            secret_seed (Optional[int], default = None):
                seed for private keys randomness

            encryption_seed (Optional[int], default = None):
                seed for encryption randomness

            initial_keys (Optional[Dict[int, LweSecretKey]] = None):
                initial keys to set before keygen
        """

        assert self._keys is not None, "Tried to generate keys on simulated client."
        self._keys.generate(
            force=force,
            secret_seed=secret_seed,
            encryption_seed=encryption_seed,
            initial_keys=initial_keys,
        )

    def encrypt(
        self,
        *args: Optional[Union[int, np.ndarray, list]],
        function_name: Optional[str] = None,
    ) -> Optional[Union[Value, tuple[Optional[Value], ...]]]:
        """
        Encrypt argument(s) to for evaluation.

        Args:
            *args (Optional[Union[int, np.ndarray, List]]):
                argument(s) for evaluation
            function_name (str):
                name of the function to encrypt

        Returns:
            Optional[Union[Value, Tuple[Optional[Value], ...]]]:
                encrypted argument(s) for evaluation
        """

        assert self._keys is not None, "Tried to encrypt on a simulated client."
        if not self._keys.are_generated:
            self._keys.generate()

        if function_name is None:
            functions = self.specs.program_info.function_list()
            if len(functions) == 1:
                function_name = functions[0]
            else:  # pragma: no cover
                msg = "The client contains more than one functions. \
Provide a `function_name` keyword argument to disambiguate."
                raise TypeError(msg)

        ordered_sanitized_args = validate_input_args(
            self._client_specs, *args, function_name=function_name
        )
        client_program = ClientProgram.create_encrypted(
            self._client_specs.program_info, self._keys._keyset  # pylint: disable=protected-access
        )
        client_circuit = client_program.get_client_circuit(function_name)

        exported = [
            (
                None
                if arg is None
                else Value(
                    client_circuit.prepare_input(
                        Value_(arg.astype(np.int64) if isinstance(arg, np.ndarray) else arg),
                        position,
                    )
                )
            )
            for position, arg in enumerate(ordered_sanitized_args)
        ]

        return tuple(exported) if len(exported) != 1 else exported[0]

    def simulate_encrypt(
        self,
        *args: Optional[Union[int, np.ndarray, list]],
        function_name: Optional[str] = None,
    ) -> Optional[Union[Value, tuple[Optional[Value], ...]]]:
        """
        Simulate encryption of argument(s) for evaluation.

        Args:
            *args (Optional[Union[int, np.ndarray, List]]):
                argument(s) for evaluation
            function_name (str):
                name of the function to encrypt

        Returns:
            Optional[Union[Value, Tuple[Optional[Value], ...]]]:
                encrypted argument(s) for evaluation
        """

        assert self._keys is None, "Tried to simulate encryption on an encrypted client."

        if function_name is None:  # pragma: no cover
            functions = self.specs.program_info.function_list()
            if len(functions) == 1:
                function_name = functions[0]
            else:
                msg = "The client contains more than one functions. \
Provide a `function_name` keyword argument to disambiguate."
                raise TypeError(msg)

        ordered_sanitized_args = validate_input_args(
            self._client_specs, *args, function_name=function_name
        )
        client_program = ClientProgram.create_simulated(self._client_specs.program_info)
        client_circuit = client_program.get_client_circuit(function_name)

        exported = [
            (
                None
                if arg is None
                else Value(
                    client_circuit.simulate_prepare_input(
                        Value_(arg.astype(np.int64) if isinstance(arg, np.ndarray) else arg),
                        position,
                    )
                )
            )
            for position, arg in enumerate(ordered_sanitized_args)
        ]

        return tuple(exported) if len(exported) != 1 else exported[0]

    def decrypt(
        self,
        *results: Union[Value, tuple[Value, ...]],
        function_name: Optional[str] = None,
    ) -> Optional[Union[int, np.ndarray, tuple[Optional[Union[int, np.ndarray]], ...]]]:
        """
        Decrypt result(s) of evaluation.

        Args:
            *results (Union[Value, Tuple[Value, ...]]):
                result(s) of evaluation
            function_name (str):
                name of the function to decrypt for

        Returns:
            Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:
                decrypted result(s) of evaluation
        """

        if function_name is None:  # pragma: no cover
            functions = self.specs.program_info.function_list()
            if len(functions) == 1:
                function_name = functions[0]
            else:  # pragma: no cover
                msg = "The client contains more than one functions. \
Provide a `function_name` keyword argument to disambiguate."
                raise TypeError(msg)

        flattened_results: list[Value] = []
        for result in results:
            if isinstance(result, tuple):  # pragma: no cover
                # this branch is impossible to cover without multiple outputs
                flattened_results.extend(result)
            else:
                flattened_results.append(result)

        assert self._keys is not None, "Tried to decrypt on a simulated client."
        assert self._keys.are_generated

        client_program = ClientProgram.create_encrypted(
            self._client_specs.program_info, self._keys._keyset  # pylint: disable=protected-access
        )
        client_circuit = client_program.get_client_circuit(function_name)

        decrypted = tuple(
            client_circuit.process_output(
                result._inner, position  # pylint: disable=protected-access
            ).to_py_val()
            for position, result in enumerate(flattened_results)
        )
        decrypted = tuple(d.astype("int64") if isinstance(d, np.ndarray) else d for d in decrypted)

        return decrypted if len(decrypted) != 1 else decrypted[0]

    def simulate_decrypt(
        self,
        *results: Union[Value, tuple[Value, ...]],
        function_name: Optional[str] = None,
    ) -> Optional[Union[int, np.ndarray, tuple[Optional[Union[int, np.ndarray]], ...]]]:
        """
        Simulate decryption of result(s) of evaluation.

        Args:
            *results (Union[Value, Tuple[Value, ...]]):
                result(s) of evaluation
            function_name (str):
                name of the function to decrypt for

        Returns:
            Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:
                decrypted result(s) of evaluation
        """

        if function_name is None:  # pragma: no cover
            functions = self.specs.program_info.function_list()
            if len(functions) == 1:
                function_name = functions[0]
            else:  # pragma: no cover
                msg = "The client contains more than one functions. \
Provide a `function_name` keyword argument to disambiguate."
                raise TypeError(msg)

        flattened_results: list[Value] = []
        for result in results:
            if isinstance(result, tuple):  # pragma: no cover
                # this branch is impossible to cover without multiple outputs
                flattened_results.extend(result)
            else:
                flattened_results.append(result)

        assert self._keys is None, "Tried to simulate decryption on an encrypted client."

        client_program = ClientProgram.create_simulated(self._client_specs.program_info)
        client_circuit = client_program.get_client_circuit(function_name)

        decrypted = tuple(
            client_circuit.simulate_process_output(
                result._inner, position  # pylint: disable=protected-access
            ).to_py_val()
            for position, result in enumerate(flattened_results)
        )
        decrypted = tuple(d.astype("int64") if isinstance(d, np.ndarray) else d for d in decrypted)

        return decrypted if len(decrypted) != 1 else decrypted[0]

    @property
    def evaluation_keys(self) -> EvaluationKeys:
        """
        Get evaluation keys for encrypted computation.

        Returns:
            EvaluationKeys
                evaluation keys for encrypted computation
        """

        assert self._keys is not None, "Tried to get evaluation keys from simulated client."
        self.keygen(force=False)
        return self._keys.evaluation
