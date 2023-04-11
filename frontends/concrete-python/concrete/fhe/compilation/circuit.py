"""
Declaration of `Circuit` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

from typing import Any, Optional, Tuple, Union

import numpy as np

# mypy: disable-error-code=attr-defined
from concrete.compiler import PublicArguments, PublicResult

from ..internal.utils import assert_that
from ..representation import Graph
from .client import Client
from .configuration import Configuration
from .keys import Keys
from .server import Server

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

        return self.client.encrypt(*args)

    def run(self, args: PublicArguments) -> PublicResult:
        """
        Evaluate circuit using encrypted arguments.

        Args:
            args (PublicArguments):
                arguments to the circuit (can be obtained with `encrypt` method of `Circuit`)

        Returns:
            PublicResult:
                encrypted result of homomorphic evaluaton
        """

        self.keygen(force=False)
        return self.server.run(args, self.client.evaluation_keys)

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

        return self.client.decrypt(result)

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
