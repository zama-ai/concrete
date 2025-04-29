"""
Declaration of `Circuit` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from concrete.compiler import CompilationContext, LweSecretKey, Parameter
from mlir.ir import Module as MlirModule

from ..representation import Graph
from .client import Client
from .configuration import Configuration
from .keys import Keys
from .module import FheFunction, FheModule
from .server import Server
from .value import Value

# pylint: enable=import-error,no-member,no-name-in-module


class Circuit:
    """
    Circuit class, to combine computation graph, mlir, client and server into a single object.
    """

    _module: FheModule
    _name: str

    def __init__(self, module: FheModule):
        assert module.function_count == 1
        self._name = next(iter(module.functions().keys()))
        self._module = module

    @property
    def _function(self) -> FheFunction:
        return getattr(self._module, self._name)

    def __str__(self):
        return self._function.graph.format()

    def draw(
        self,
        *,
        horizontal: bool = False,
        save_to: Optional[Union[Path, str]] = None,
        show: bool = False,
    ) -> Path:
        """
        Draw the graph of the circuit.

        That this function requires the python `pygraphviz` package
        which itself requires the installation of `graphviz` packages

        (see https://pygraphviz.github.io/documentation/stable/install.html)

        Args:
            horizontal (bool, default = False):
                whether to draw horizontally

            save_to (Optional[Path], default = None):
                path to save the drawing
                a temporary file will be used if it's None

            show (bool, default = False):
                whether to show the drawing using matplotlib

        Returns:
            Path:
                path to the drawing
        """

        return self._function.graph.draw(horizontal=horizontal, save_to=save_to, show=show)

    @property
    def mlir(self) -> str:
        """Textual representation of the MLIR module.

        Returns:
            str: textual representation of the MLIR module
        """
        return str(self._module.mlir_module).strip()

    def enable_fhe_simulation(self):
        """
        Enable FHE simulation.
        """
        self._module.simulation_runtime.init()

    def enable_fhe_execution(self):
        """
        Enable FHE execution.
        """
        self._module.execution_runtime.init()  # pragma: no cover

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

        return self._function.simulate(*args)

    @property
    def keys(self) -> Optional[Keys]:
        """
        Get the keys of the circuit.
        """
        return self._module.keys

    @keys.setter
    def keys(self, new_keys: Keys):
        """
        Set the keys of the circuit.
        """
        self._module.keys = new_keys

    def keygen(
        self,
        force: bool = False,
        seed: Optional[int] = None,
        encryption_seed: Optional[int] = None,
        initial_keys: Optional[dict[int, LweSecretKey]] = None,
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

            initial_keys (Optional[Dict[int, LweSecretKey]] = None):
                initial keys to set before keygen
        """
        self._module.keygen(
            force=force,
            seed=seed,
            encryption_seed=encryption_seed,
            initial_keys=initial_keys,
        )

    def encrypt(
        self,
        *args: Optional[Union[int, np.ndarray, list]],
    ) -> Optional[Union[Value, tuple[Optional[Value], ...]]]:
        """
        Encrypt argument(s) to for evaluation.

        Args:
            *args (Optional[Union[int, numpy.ndarray, List]]):
                argument(s) for evaluation

        Returns:
            Optional[Union[Value, Tuple[Optional[Value], ...]]]:
                encrypted argument(s) for evaluation
        """
        return self._function.encrypt(*args)

    def run(
        self,
        *args: Optional[Union[Value, tuple[Optional[Value], ...]]],
    ) -> Union[Value, tuple[Value, ...]]:
        """
        Evaluate the circuit.

        Args:
            *args (Value):
                argument(s) for evaluation

        Returns:
            Union[Value, Tuple[Value, ...]]:
                result(s) of evaluation
        """

        return self._function.run_sync(*args)

    def decrypt(
        self,
        *results: Union[Value, tuple[Value, ...]],
    ) -> Optional[Union[int, np.ndarray, tuple[Optional[Union[int, np.ndarray]], ...]]]:
        """
        Decrypt result(s) of evaluation.

        Args:
            *results (Union[Value, Tuple[Value, ...]]):
                result(s) of evaluation

        Returns:
            Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:
                decrypted result(s) of evaluation
        """

        return self._function.decrypt(*results)

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

        return self._function.encrypt_run_decrypt(*args)

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """

        self._module.cleanup()

    # Properties

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the circuit.
        """
        return self._module.size_of_secret_keys  # pragma: no cover

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the circuit.
        """
        return self._module.size_of_bootstrap_keys  # pragma: no cover

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the circuit.
        """
        return self._module.size_of_keyswitch_keys  # pragma: no cover

    @property
    def size_of_inputs(self) -> int:
        """
        Get size of the inputs of the circuit.
        """
        return self._function.size_of_inputs  # pragma: no cover

    @property
    def size_of_outputs(self) -> int:
        """
        Get size of the outputs of the circuit.
        """
        return self._function.size_of_outputs  # pragma: no cover

    @property
    def p_error(self) -> float:
        """
        Get probability of error for each simple TLU (on a scalar).
        """
        return self._module.p_error  # pragma: no cover

    @property
    def global_p_error(self) -> float:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """
        return self._module.p_error  # pragma: no cover

    @property
    def complexity(self) -> float:
        """
        Get complexity of the circuit.
        """
        return self._module.complexity  # pragma: no cover

    @property
    def memory_usage_per_location(self) -> dict[str, Optional[int]]:
        """
        Get the memory usage of operations in the circuit per location.
        """
        return self._function.execution_runtime.val.server.memory_usage_per_location(
            self._name
        )  # pragma: no cover

    # Programmable Bootstrap Statistics

    @property
    def programmable_bootstrap_count(self) -> int:
        """
        Get the number of programmable bootstraps in the circuit.
        """
        return self._function.programmable_bootstrap_count  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of programmable bootstraps per bit width in the circuit.
        """
        return self._function.programmable_bootstrap_count_per_parameter  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of programmable bootstraps per tag in the circuit.
        """
        return self._function.programmable_bootstrap_count_per_tag  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[int, int]]:
        """
        Get the number of programmable bootstraps per tag per bit width in the circuit.
        """
        return self._function.programmable_bootstrap_count_per_tag_per_parameter  # pragma: no cover

    # Key Switch Statistics

    @property
    def key_switch_count(self) -> int:
        """
        Get the number of key switches in the circuit.
        """
        return self._function.key_switch_count  # pragma: no cover

    @property
    def key_switch_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of key switches per parameter in the circuit.
        """
        return self._function.key_switch_count_per_parameter  # pragma: no cover

    @property
    def key_switch_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of key switches per tag in the circuit.
        """
        return self._function.key_switch_count_per_tag  # pragma: no cover

    @property
    def key_switch_count_per_tag_per_parameter(self) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of key switches per tag per parameter in the circuit.
        """
        return self._function.key_switch_count_per_tag_per_parameter  # pragma: no cover

    # Packing Key Switch Statistics

    @property
    def packing_key_switch_count(self) -> int:
        """
        Get the number of packing key switches in the circuit.
        """
        return self._function.packing_key_switch_count  # pragma: no cover

    @property
    def packing_key_switch_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of packing key switches per parameter in the circuit.
        """
        return self._function.packing_key_switch_count_per_parameter  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of packing key switches per tag in the circuit.
        """
        return self._function.packing_key_switch_count_per_tag  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of packing key switches per tag per parameter in the circuit.
        """
        return self._function.packing_key_switch_count_per_tag_per_parameter  # pragma: no cover

    # Clear Addition Statistics

    @property
    def clear_addition_count(self) -> int:
        """
        Get the number of clear additions in the circuit.
        """
        return self._function.clear_addition_count  # pragma: no cover

    @property
    def clear_addition_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of clear additions per parameter in the circuit.
        """
        return self._function.clear_addition_count_per_parameter  # pragma: no cover

    @property
    def clear_addition_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of clear additions per tag in the circuit.
        """
        return self._function.clear_addition_count_per_tag  # pragma: no cover

    @property
    def clear_addition_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of clear additions per tag per parameter in the circuit.
        """
        return self._function.clear_addition_count_per_tag_per_parameter  # pragma: no cover

    # Encrypted Addition Statistics

    @property
    def encrypted_addition_count(self) -> int:
        """
        Get the number of encrypted additions in the circuit.
        """
        return self._function.encrypted_addition_count  # pragma: no cover

    @property
    def encrypted_addition_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of encrypted additions per parameter in the circuit.
        """
        return self._function.encrypted_addition_count_per_parameter  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of encrypted additions per tag in the circuit.
        """
        return self._function.encrypted_addition_count_per_tag  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of encrypted additions per tag per parameter in the circuit.
        """
        return self._function.encrypted_addition_count_per_tag_per_parameter  # pragma: no cover

    # Clear Multiplication Statistics

    @property
    def clear_multiplication_count(self) -> int:
        """
        Get the number of clear multiplications in the circuit.
        """
        return self._function.clear_multiplication_count  # pragma: no cover

    @property
    def clear_multiplication_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of clear multiplications per parameter in the circuit.
        """
        return self._function.clear_multiplication_count_per_parameter  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of clear multiplications per tag in the circuit.
        """
        return self._function.clear_multiplication_count_per_tag  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of clear multiplications per tag per parameter in the circuit.
        """
        return self._function.clear_multiplication_count_per_tag_per_parameter  # pragma: no cover

    # Encrypted Negation Statistics

    @property
    def encrypted_negation_count(self) -> int:
        """
        Get the number of encrypted negations in the circuit.
        """
        return self._function.encrypted_negation_count  # pragma: no cover

    @property
    def encrypted_negation_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of encrypted negations per parameter in the circuit.
        """
        return self._function.encrypted_negation_count_per_parameter  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of encrypted negations per tag in the circuit.
        """
        return self._function.encrypted_negation_count_per_tag  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of encrypted negations per tag per parameter in the circuit.
        """
        return self._function.encrypted_negation_count_per_tag_per_parameter  # pragma: no cover

    # All Statistics

    @property
    def statistics(self) -> dict:
        """
        Get all statistics of the circuit.
        """
        mod_stats = self._module.statistics
        func_stats = mod_stats.pop("functions")[self._name]
        return {**mod_stats, **func_stats}

    @property
    def configuration(self) -> Configuration:
        """
        Return the circuit configuration.
        """
        return self._module.configuration

    @property
    def graph(self) -> Graph:
        """
        Return the circuit graph.
        """
        return self._function.graph

    @property
    def mlir_module(self) -> MlirModule:
        """
        Return the circuit mlir module.
        """
        return self._module.mlir_module

    @property
    def compilation_context(self) -> CompilationContext:
        """
        Return the circuit compilation context.
        """
        return self._module.compilation_context

    @property
    def client(self) -> Client:
        """
        Return the circuit client.
        """
        return self._module.client

    @property
    def server(self) -> Server:
        """
        Return the circuit server.
        """
        return self._module.server

    @property
    def simulator(self) -> Server:
        """
        Return the circuit simulator.
        """
        return self._module.simulator
