"""
Declaration of `Circuit` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from concrete.compiler import SimulatedValueDecrypter, SimulatedValueExporter

from ..internal.utils import assert_that
from ..representation import Graph
from .client import Client
from .configuration import Configuration
from .keys import Keys
from .server import Server
from .utils import validate_input_args
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
    simulator: Server

    def __init__(self, graph: Graph, mlir: str, configuration: Optional[Configuration] = None):
        self.configuration = configuration if configuration is not None else Configuration()

        self.graph = graph
        self.mlir = mlir

        if self.configuration.fhe_simulation:
            self.enable_fhe_simulation()

        if self.configuration.fhe_execution:
            self.enable_fhe_execution()

    def __str__(self):
        return self.graph.format()

    def enable_fhe_simulation(self):
        """
        Enable FHE simulation.
        """

        if not hasattr(self, "simulator"):
            self.simulator = Server.create(self.mlir, self.configuration, is_simulated=True)

    def enable_fhe_execution(self):
        """
        Enable FHE execution.
        """

        if not hasattr(self, "server"):
            self.server = Server.create(self.mlir, self.configuration)

            keyset_cache_directory = None
            if self.configuration.use_insecure_key_cache:
                assert_that(self.configuration.enable_unsafe_features)
                assert_that(self.configuration.insecure_key_cache_location is not None)
                keyset_cache_directory = self.configuration.insecure_key_cache_location

            self.client = Client(self.server.client_specs, keyset_cache_directory)

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

        if not hasattr(self, "simulator"):  # pragma: no cover
            self.enable_fhe_simulation()

        ordered_validated_args = validate_input_args(self.simulator.client_specs, *args)

        exporter = SimulatedValueExporter.new(self.simulator.client_specs.client_parameters)
        exported = [
            None
            if arg is None
            else Value(
                exporter.export_tensor(position, arg.flatten().tolist(), list(arg.shape))
                if isinstance(arg, np.ndarray) and arg.shape != ()
                else exporter.export_scalar(position, int(arg))
            )
            for position, arg in enumerate(ordered_validated_args)
        ]

        results = self.simulator.run(*exported)
        if not isinstance(results, tuple):
            results = (results,)

        decrypter = SimulatedValueDecrypter.new(self.simulator.client_specs.client_parameters)
        decrypted = tuple(
            decrypter.decrypt(position, result.inner) for position, result in enumerate(results)
        )

        return decrypted if len(decrypted) != 1 else decrypted[0]

    @property
    def keys(self) -> Keys:
        """
        Get the keys of the circuit.
        """

        if not hasattr(self, "client"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.client.keys

    @keys.setter
    def keys(self, new_keys: Keys):
        """
        Set the keys of the circuit.
        """

        if not hasattr(self, "client"):  # pragma: no cover
            self.enable_fhe_execution()

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

        if not hasattr(self, "client"):  # pragma: no cover
            self.enable_fhe_execution()

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

        if not hasattr(self, "client"):  # pragma: no cover
            self.enable_fhe_execution()

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

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

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

        if not hasattr(self, "client"):  # pragma: no cover
            self.enable_fhe_execution()

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

        if hasattr(self, "server"):  # pragma: no cover
            self.server.cleanup()

    @property
    def complexity(self) -> float:
        """
        Get complexity of the circuit.
        """

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.server.complexity

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the circuit.
        """

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.server.size_of_secret_keys

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the circuit.
        """

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.server.size_of_bootstrap_keys

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the circuit.
        """

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.server.size_of_keyswitch_keys

    @property
    def size_of_inputs(self) -> int:
        """
        Get size of the inputs of the circuit.
        """

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.server.size_of_inputs

    @property
    def size_of_outputs(self) -> int:
        """
        Get size of the outputs of the circuit.
        """

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.server.size_of_outputs

    @property
    def p_error(self) -> int:
        """
        Get probability of error for each simple TLU (on a scalar).
        """

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.server.p_error

    @property
    def global_p_error(self) -> int:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """

        if not hasattr(self, "server"):  # pragma: no cover
            self.enable_fhe_execution()

        return self.server.global_p_error
