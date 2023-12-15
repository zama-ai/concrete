"""
Declaration of `Circuit` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from concrete.compiler import (
    CompilationContext,
    Parameter,
    SimulatedValueDecrypter,
    SimulatedValueExporter,
)
from mlir.ir import Module as MlirModule

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
    mlir_module: MlirModule
    compilation_context: CompilationContext

    client: Client
    server: Server
    simulator: Server

    def __init__(
        self,
        graph: Graph,
        mlir: MlirModule,
        compilation_context: CompilationContext,
        configuration: Optional[Configuration] = None,
    ):
        self.configuration = configuration if configuration is not None else Configuration()

        self.graph = graph
        self.mlir_module = mlir
        self.compilation_context = compilation_context

        if self.configuration.fhe_simulation:
            self.enable_fhe_simulation()

        if self.configuration.fhe_execution:
            self.enable_fhe_execution()

    def __str__(self):
        return self.graph.format()

    @property
    def mlir(self) -> str:
        """Textual representation of the MLIR module.

        Returns:
            str: textual representation of the MLIR module
        """
        return str(self.mlir_module).strip()

    def enable_fhe_simulation(self):
        """
        Enable FHE simulation.
        """

        if not hasattr(self, "simulator"):
            self.simulator = Server.create(
                self.mlir_module,
                self.configuration,
                is_simulated=True,
                compilation_context=self.compilation_context,
            )

    def enable_fhe_execution(self):
        """
        Enable FHE execution.
        """

        if not hasattr(self, "server"):
            self.server = Server.create(
                self.mlir_module, self.configuration, compilation_context=self.compilation_context
            )

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

        if not hasattr(self, "client"):  # pragma: no cover
            self.enable_fhe_execution()

        self.client.keygen(force, seed, encryption_seed)

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

    # Properties

    def _property(self, name: str) -> Any:
        """
        Get a property of the circuit by name.

        Args:
            name (str):
                name of the property

        Returns:
            Any:
                statistic
        """

        if hasattr(self, "simulator"):
            return getattr(self.simulator, name)  # pragma: no cover

        if not hasattr(self, "server"):
            self.enable_fhe_execution()  # pragma: no cover

        return getattr(self.server, name)

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the circuit.
        """
        return self._property("size_of_secret_keys")  # pragma: no cover

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the circuit.
        """
        return self._property("size_of_bootstrap_keys")  # pragma: no cover

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the circuit.
        """
        return self._property("size_of_keyswitch_keys")  # pragma: no cover

    @property
    def size_of_inputs(self) -> int:
        """
        Get size of the inputs of the circuit.
        """
        return self._property("size_of_inputs")  # pragma: no cover

    @property
    def size_of_outputs(self) -> int:
        """
        Get size of the outputs of the circuit.
        """
        return self._property("size_of_outputs")  # pragma: no cover

    @property
    def p_error(self) -> int:
        """
        Get probability of error for each simple TLU (on a scalar).
        """
        return self._property("p_error")  # pragma: no cover

    @property
    def global_p_error(self) -> int:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """
        return self._property("global_p_error")  # pragma: no cover

    @property
    def complexity(self) -> float:
        """
        Get complexity of the circuit.
        """
        return self._property("complexity")  # pragma: no cover

    # Programmable Bootstrap Statistics

    @property
    def programmable_bootstrap_count(self) -> int:
        """
        Get the number of programmable bootstraps in the circuit.
        """
        return self._property("programmable_bootstrap_count")  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of programmable bootstraps per bit width in the circuit.
        """
        return self._property("programmable_bootstrap_count_per_parameter")  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of programmable bootstraps per tag in the circuit.
        """
        return self._property("programmable_bootstrap_count_per_tag")  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag_per_parameter(self) -> Dict[str, Dict[int, int]]:
        """
        Get the number of programmable bootstraps per tag per bit width in the circuit.
        """
        return self._property(
            "programmable_bootstrap_count_per_tag_per_parameter"
        )  # pragma: no cover

    # Key Switch Statistics

    @property
    def key_switch_count(self) -> int:
        """
        Get the number of key switches in the circuit.
        """
        return self._property("key_switch_count")  # pragma: no cover

    @property
    def key_switch_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of key switches per parameter in the circuit.
        """
        return self._property("key_switch_count_per_parameter")  # pragma: no cover

    @property
    def key_switch_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of key switches per tag in the circuit.
        """
        return self._property("key_switch_count_per_tag")  # pragma: no cover

    @property
    def key_switch_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of key switches per tag per parameter in the circuit.
        """
        return self._property("key_switch_count_per_tag_per_parameter")  # pragma: no cover

    # Packing Key Switch Statistics

    @property
    def packing_key_switch_count(self) -> int:
        """
        Get the number of packing key switches in the circuit.
        """
        return self._property("packing_key_switch_count")  # pragma: no cover

    @property
    def packing_key_switch_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of packing key switches per parameter in the circuit.
        """
        return self._property("packing_key_switch_count_per_parameter")  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of packing key switches per tag in the circuit.
        """
        return self._property("packing_key_switch_count_per_tag")  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of packing key switches per tag per parameter in the circuit.
        """
        return self._property("packing_key_switch_count_per_tag_per_parameter")  # pragma: no cover

    # Clear Addition Statistics

    @property
    def clear_addition_count(self) -> int:
        """
        Get the number of clear additions in the circuit.
        """
        return self._property("clear_addition_count")  # pragma: no cover

    @property
    def clear_addition_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of clear additions per parameter in the circuit.
        """
        return self._property("clear_addition_count_per_parameter")  # pragma: no cover

    @property
    def clear_addition_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of clear additions per tag in the circuit.
        """
        return self._property("clear_addition_count_per_tag")  # pragma: no cover

    @property
    def clear_addition_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of clear additions per tag per parameter in the circuit.
        """
        return self._property("clear_addition_count_per_tag_per_parameter")  # pragma: no cover

    # Encrypted Addition Statistics

    @property
    def encrypted_addition_count(self) -> int:
        """
        Get the number of encrypted additions in the circuit.
        """
        return self._property("encrypted_addition_count")  # pragma: no cover

    @property
    def encrypted_addition_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of encrypted additions per parameter in the circuit.
        """
        return self._property("encrypted_addition_count_per_parameter")  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of encrypted additions per tag in the circuit.
        """
        return self._property("encrypted_addition_count_per_tag")  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of encrypted additions per tag per parameter in the circuit.
        """
        return self._property("encrypted_addition_count_per_tag_per_parameter")  # pragma: no cover

    # Clear Multiplication Statistics

    @property
    def clear_multiplication_count(self) -> int:
        """
        Get the number of clear multiplications in the circuit.
        """
        return self._property("clear_multiplication_count")  # pragma: no cover

    @property
    def clear_multiplication_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of clear multiplications per parameter in the circuit.
        """
        return self._property("clear_multiplication_count_per_parameter")  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of clear multiplications per tag in the circuit.
        """
        return self._property("clear_multiplication_count_per_tag")  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of clear multiplications per tag per parameter in the circuit.
        """
        return self._property(
            "clear_multiplication_count_per_tag_per_parameter"
        )  # pragma: no cover

    # Encrypted Negation Statistics

    @property
    def encrypted_negation_count(self) -> int:
        """
        Get the number of encrypted negations in the circuit.
        """
        return self._property("encrypted_negation_count")  # pragma: no cover

    @property
    def encrypted_negation_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of encrypted negations per parameter in the circuit.
        """
        return self._property("encrypted_negation_count_per_parameter")  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of encrypted negations per tag in the circuit.
        """
        return self._property("encrypted_negation_count_per_tag")  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of encrypted negations per tag per parameter in the circuit.
        """
        return self._property("encrypted_negation_count_per_tag_per_parameter")  # pragma: no cover

    # All Statistics

    @property
    def statistics(self) -> Dict:
        """
        Get all statistics of the circuit.
        """
        return self._property("statistics")  # pragma: no cover
