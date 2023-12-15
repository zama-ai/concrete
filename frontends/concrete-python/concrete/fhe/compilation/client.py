"""
Declaration of `Client` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from concrete.compiler import EvaluationKeys, ValueDecrypter, ValueExporter

from .keys import Keys
from .specs import ClientSpecs
from .utils import validate_input_args
from .value import Value

# pylint: enable=import-error,no-member,no-name-in-module


class Client:
    """
    Client class, which can be used to manage keys, encrypt arguments and decrypt results.
    """

    specs: ClientSpecs
    _keys: Keys

    def __init__(
        self,
        client_specs: ClientSpecs,
        keyset_cache_directory: Optional[Union[str, Path]] = None,
    ):
        self.specs = client_specs
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
            with open(Path(tmp_dir) / "client.specs.json", "rb") as f:
                client_specs = ClientSpecs.deserialize(f.read())

        return Client(client_specs, keyset_cache_directory)

    @property
    def keys(self) -> Keys:
        """
        Get the keys for the client.
        """
        return self._keys

    @keys.setter
    def keys(self, new_keys: Keys):
        """
        Set the keys for the client.
        """
        # TODO: implement verification for compatibility with keyset.

        self._keys = new_keys

    def keygen(
        self, force: bool = False, seed: Optional[int] = None, encryption_seed: Optional[int] = None
    ):
        """
        Generate keys required for homomorphic evaluation.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated

            seed (Optional[int], default = None):
                seed for private keys randomness

            encryption_seed (Optional[int], default = None):
                seed for encryption randomness
        """

        self.keys.generate(force=force, seed=seed, encryption_seed=encryption_seed)

    def encrypt(
        self,
        *args: Optional[Union[int, np.ndarray, List]],
    ) -> Optional[Union[Value, Tuple[Optional[Value], ...]]]:
        """
        Encrypt argument(s) to for evaluation.

        Args:
            *args (Optional[Union[int, np.ndarray, List]]):
                argument(s) for evaluation

        Returns:
            Optional[Union[Value, Tuple[Optional[Value], ...]]]:
                encrypted argument(s) for evaluation
        """

        ordered_sanitized_args = validate_input_args(self.specs, *args)

        self.keygen(force=False)
        keyset = self.keys._keyset  # pylint: disable=protected-access

        exporter = ValueExporter.new(keyset, self.specs.client_parameters)
        exported = [
            None
            if arg is None
            else Value(
                exporter.export_tensor(position, arg.flatten().tolist(), list(arg.shape))
                if isinstance(arg, np.ndarray) and arg.shape != ()
                else exporter.export_scalar(position, int(arg))
            )
            for position, arg in enumerate(ordered_sanitized_args)
        ]

        return tuple(exported) if len(exported) != 1 else exported[0]

    def decrypt(
        self,
        *results: Union[Value, Tuple[Value, ...]],
    ) -> Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:
        """
        Decrypt result(s) of evaluation.

        Args:
            *results (Union[Value, Tuple[Value, ...]]):
                result(s) of evaluation

        Returns:
            Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:
                decrypted result(s) of evaluation
        """

        flattened_results: List[Value] = []
        for result in results:
            if isinstance(result, tuple):  # pragma: no cover
                # this branch is impossible to cover without multiple outputs
                flattened_results.extend(result)
            else:
                flattened_results.append(result)

        self.keygen(force=False)
        keyset = self.keys._keyset  # pylint: disable=protected-access

        decrypter = ValueDecrypter.new(keyset, self.specs.client_parameters)
        decrypted = tuple(
            decrypter.decrypt(position, result.inner)
            for position, result in enumerate(flattened_results)
        )

        return decrypted if len(decrypted) != 1 else decrypted[0]

    @property
    def evaluation_keys(self) -> EvaluationKeys:
        """
        Get evaluation keys for encrypted computation.

        Returns:
            EvaluationKeys
                evaluation keys for encrypted computation
        """

        self.keygen(force=False)
        return self.keys.evaluation
