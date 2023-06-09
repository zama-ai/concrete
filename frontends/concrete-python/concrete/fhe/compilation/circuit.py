"""
Declaration of `Circuit` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from ..internal.utils import assert_that
from ..representation import Graph
from .client import Client
from .configuration import Configuration
from .keys import Keys
from .server import Server
from .value import Value

# pylint: enable=import-error,no-member,no-name-in-module


class Circuit:
    """
    Circuit class, to combine computation graph, mlir, client and server into a single object.
    """

    configuration: Configuration

    graph: Graph
    mlir: str

    client: Client
    server: Server

    def __init__(self, graph: Graph, mlir: str, configuration: Optional[Configuration] = None):
        self.configuration = configuration if configuration is not None else Configuration()

        self.graph = graph
        self.mlir = mlir

        self._initialize_client_and_server()

    def _initialize_client_and_server(self):
        self.server = Server.create(self.mlir, self.configuration)

        keyset_cache_directory = None
        if self.configuration.use_insecure_key_cache:
            assert_that(self.configuration.enable_unsafe_features)
            assert_that(self.configuration.insecure_key_cache_location is not None)
            keyset_cache_directory = self.configuration.insecure_key_cache_location

        self.client = Client(self.server.client_specs, keyset_cache_directory)

    def __str__(self):
        return self.graph.format()

    def simulate(self, *args: Any) -> Any:
        """
        Simulate execution of the circuit.

        Args:
            *args (Any):
                inputs to the circuit

        Returns:
            Any:
                result of the simulation
        """

        return self.graph(*args, p_error=self.p_error)

    @property
    def keys(self) -> Keys:
        """
        Get the keys of the circuit.
        """
        return self.client.keys

    @keys.setter
    def keys(self, new_keys: Keys):
        """
        Set the keys of the circuit.
        """
        self.client.keys = new_keys

    def keygen(self, force: bool = False, seed: Optional[int] = None):
        """
        Generate keys required for homomorphic evaluation.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated

            seed (Optional[int], default = None):
                seed for randomness
        """

        self.client.keygen(force, seed)

    def encrypt(
        self,
        *args: Optional[Union[int, np.ndarray, List]],
    ) -> Optional[Union[Value, Tuple[Optional[Value], ...]]]:
        """
        Encrypt argument(s) to for evaluation.

        Args:
            *args (Optional[Union[int, numpy.ndarray, List]]):
                argument(s) for evaluation

        Returns:
            Optional[Union[Value, Tuple[Optional[Value], ...]]]:
                encrypted argument(s) for evaluation
        """

        return self.client.encrypt(*args)

    def run(
        self,
        *args: Optional[Union[Value, Tuple[Optional[Value], ...]]],
    ) -> Union[Value, Tuple[Value, ...]]:
        """
        Evaluate the circuit.

        Args:
            *args (Value):
                argument(s) for evaluation

        Returns:
            Union[Value, Tuple[Value, ...]]:
                result(s) of evaluation
        """

        self.keygen(force=False)
        return self.server.run(*args, evaluation_keys=self.client.evaluation_keys)

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

        return self.client.decrypt(*results)

    def encrypt_run_decrypt(self, *args: Any) -> Any:
        """
        Encrypt inputs, run the circuit, and decrypt the outputs in one go.

        Args:
            *args (Union[int, numpy.ndarray]):
                inputs to the circuit

        Returns:
            Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
                clear result of homomorphic evaluation
        """

        return self.decrypt(self.run(self.encrypt(*args)))

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """

        self.server.cleanup()

    @property
    def complexity(self) -> float:
        """
        Get complexity of the circuit.
        """
        return self.server.complexity

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the circuit.
        """
        return self.server.size_of_secret_keys

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the circuit.
        """
        return self.server.size_of_bootstrap_keys

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the circuit.
        """
        return self.server.size_of_keyswitch_keys

    @property
    def size_of_inputs(self) -> int:
        """
        Get size of the inputs of the circuit.
        """
        return self.server.size_of_inputs

    @property
    def size_of_outputs(self) -> int:
        """
        Get size of the outputs of the circuit.
        """
        return self.server.size_of_outputs

    @property
    def p_error(self) -> int:
        """
        Get probability of error for each simple TLU (on a scalar).
        """
        return self.server.p_error

    @property
    def global_p_error(self) -> int:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """
        return self.server.global_p_error
