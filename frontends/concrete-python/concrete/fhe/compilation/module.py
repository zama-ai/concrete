"""
Declaration of `FheModule` classes.
"""

# pylint: disable=import-error,no-member,no-name-in-module

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

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


class ExecutionRt(NamedTuple):
    """
    Runtime object class for execution.
    """

    client: Client
    server: Server


class SimulationRt(NamedTuple):
    """
    Runtime object class for simulation.
    """

    server: Server


class FheFunction:
    """
    Fhe function class, allowing to run or simulate one function of an fhe module.
    """

    runtime: Union[ExecutionRt, SimulationRt]
    graph: Graph
    name: str

    def __init__(self, name: str, runtime: Union[ExecutionRt, SimulationRt], graph: Graph):
        self.name = name
        self.runtime = runtime
        self.graph = graph

    def __call__(
        self,
        *args: Any,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        Tuple[Union[np.bool_, np.integer, np.floating, np.ndarray], ...],
    ]:
        return self.graph(*args)

    def draw(
        self,
        *,
        horizontal: bool = False,
        save_to: Optional[Union[Path, str]] = None,
        show: bool = False,
    ) -> Path:
        """
        Draw the graph of the function.

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
        return self.graph.draw(horizontal=horizontal, save_to=save_to, show=show)

    def __str__(self):
        return self.graph.format()

    def simulate(self, *args: Any) -> Any:
        """
        Simulate execution of the function.

        Args:
            *args (Any):
                inputs to the function

        Returns:
            Any:
                result of the simulation
        """
        assert isinstance(self.runtime, SimulationRt)

        ordered_validated_args = validate_input_args(
            self.runtime.server.client_specs, *args, function_name=self.name
        )

        exporter = SimulatedValueExporter.new(
            self.runtime.server.client_specs.client_parameters, self.name
        )
        exported = [
            (
                None
                if arg is None
                else Value(
                    exporter.export_tensor(position, arg.flatten().tolist(), list(arg.shape))
                    if isinstance(arg, np.ndarray) and arg.shape != ()
                    else exporter.export_scalar(position, int(arg))
                )
            )
            for position, arg in enumerate(ordered_validated_args)
        ]

        results = self.runtime.server.run(*exported, function_name=self.name)
        if not isinstance(results, tuple):
            results = (results,)

        decrypter = SimulatedValueDecrypter.new(
            self.runtime.server.client_specs.client_parameters, self.name
        )
        decrypted = tuple(
            decrypter.decrypt(position, result.inner) for position, result in enumerate(results)
        )

        return decrypted if len(decrypted) != 1 else decrypted[0]

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
        assert isinstance(self.runtime, ExecutionRt)
        return self.runtime.client.encrypt(*args, function_name=self.name)

    def run(
        self,
        *args: Optional[Union[Value, Tuple[Optional[Value], ...]]],
    ) -> Union[Value, Tuple[Value, ...]]:
        """
        Evaluate the function.

        Args:
            *args (Value):
                argument(s) for evaluation

        Returns:
            Union[Value, Tuple[Value, ...]]:
                result(s) of evaluation
        """
        assert isinstance(self.runtime, ExecutionRt)
        return self.runtime.server.run(
            *args, evaluation_keys=self.runtime.client.evaluation_keys, function_name=self.name
        )

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
        assert isinstance(self.runtime, ExecutionRt)
        return self.runtime.client.decrypt(*results, function_name=self.name)

    def encrypt_run_decrypt(self, *args: Any) -> Any:
        """
        Encrypt inputs, run the function, and decrypt the outputs in one go.

        Args:
            *args (Union[int, numpy.ndarray]):
                inputs to the function

        Returns:
            Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
                clear result of homomorphic evaluation
        """
        return self.decrypt(self.run(self.encrypt(*args)))

    @property
    def size_of_inputs(self) -> int:
        """
        Get size of the inputs of the function.
        """
        return self.runtime.server.size_of_inputs(self.name)  # pragma: no cover

    @property
    def size_of_outputs(self) -> int:
        """
        Get size of the outputs of the function.
        """
        return self.runtime.server.size_of_outputs(self.name)  # pragma: no cover

    # Programmable Bootstrap Statistics

    @property
    def programmable_bootstrap_count(self) -> int:
        """
        Get the number of programmable bootstraps in the function.
        """
        return self.runtime.server.programmable_bootstrap_count(self.name)  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of programmable bootstraps per bit width in the function.
        """
        return self.runtime.server.programmable_bootstrap_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of programmable bootstraps per tag in the function.
        """
        return self.runtime.server.programmable_bootstrap_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag_per_parameter(self) -> Dict[str, Dict[int, int]]:
        """
        Get the number of programmable bootstraps per tag per bit width in the function.
        """
        return self.runtime.server.programmable_bootstrap_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Key Switch Statistics

    @property
    def key_switch_count(self) -> int:
        """
        Get the number of key switches in the function.
        """
        return self.runtime.server.key_switch_count(self.name)  # pragma: no cover

    @property
    def key_switch_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of key switches per parameter in the function.
        """
        return self.runtime.server.key_switch_count_per_parameter(self.name)  # pragma: no cover

    @property
    def key_switch_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of key switches per tag in the function.
        """
        return self.runtime.server.key_switch_count_per_tag(self.name)  # pragma: no cover

    @property
    def key_switch_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of key switches per tag per parameter in the function.
        """
        return self.runtime.server.key_switch_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Packing Key Switch Statistics

    @property
    def packing_key_switch_count(self) -> int:
        """
        Get the number of packing key switches in the function.
        """
        return self.runtime.server.packing_key_switch_count(self.name)  # pragma: no cover

    @property
    def packing_key_switch_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of packing key switches per parameter in the function.
        """
        return self.runtime.server.packing_key_switch_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of packing key switches per tag in the function.
        """
        return self.runtime.server.packing_key_switch_count_per_tag(self.name)  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of packing key switches per tag per parameter in the function.
        """
        return self.runtime.server.packing_key_switch_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Clear Addition Statistics

    @property
    def clear_addition_count(self) -> int:
        """
        Get the number of clear additions in the function.
        """
        return self.runtime.server.clear_addition_count(self.name)  # pragma: no cover

    @property
    def clear_addition_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of clear additions per parameter in the function.
        """
        return self.runtime.server.clear_addition_count_per_parameter(self.name)  # pragma: no cover

    @property
    def clear_addition_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of clear additions per tag in the function.
        """
        return self.runtime.server.clear_addition_count_per_tag(self.name)  # pragma: no cover

    @property
    def clear_addition_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of clear additions per tag per parameter in the function.
        """
        return self.runtime.server.clear_addition_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Encrypted Addition Statistics

    @property
    def encrypted_addition_count(self) -> int:
        """
        Get the number of encrypted additions in the function.
        """
        return self.runtime.server.encrypted_addition_count(self.name)  # pragma: no cover

    @property
    def encrypted_addition_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of encrypted additions per parameter in the function.
        """
        return self.runtime.server.encrypted_addition_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of encrypted additions per tag in the function.
        """
        return self.runtime.server.encrypted_addition_count_per_tag(self.name)  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of encrypted additions per tag per parameter in the function.
        """
        return self.runtime.server.encrypted_addition_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Clear Multiplication Statistics

    @property
    def clear_multiplication_count(self) -> int:
        """
        Get the number of clear multiplications in the function.
        """
        return self.runtime.server.clear_multiplication_count(self.name)  # pragma: no cover

    @property
    def clear_multiplication_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of clear multiplications per parameter in the function.
        """
        return self.runtime.server.clear_multiplication_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of clear multiplications per tag in the function.
        """
        return self.runtime.server.clear_multiplication_count_per_tag(self.name)  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of clear multiplications per tag per parameter in the function.
        """
        return self.runtime.server.clear_multiplication_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Encrypted Negation Statistics

    @property
    def encrypted_negation_count(self) -> int:
        """
        Get the number of encrypted negations in the function.
        """
        return self.runtime.server.encrypted_negation_count(self.name)  # pragma: no cover

    @property
    def encrypted_negation_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of encrypted negations per parameter in the function.
        """
        return self.runtime.server.encrypted_negation_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of encrypted negations per tag in the function.
        """
        return self.runtime.server.encrypted_negation_count_per_tag(self.name)  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of encrypted negations per tag per parameter in the function.
        """
        return self.runtime.server.encrypted_negation_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def statistics(self) -> Dict:
        """
        Get all statistics of the function.
        """
        attributes = [
            "size_of_inputs",
            "size_of_outputs",
            "programmable_bootstrap_count",
            "programmable_bootstrap_count_per_parameter",
            "programmable_bootstrap_count_per_tag",
            "programmable_bootstrap_count_per_tag_per_parameter",
            "key_switch_count",
            "key_switch_count_per_parameter",
            "key_switch_count_per_tag",
            "key_switch_count_per_tag_per_parameter",
            "packing_key_switch_count",
            "packing_key_switch_count_per_parameter",
            "packing_key_switch_count_per_tag",
            "packing_key_switch_count_per_tag_per_parameter",
            "clear_addition_count",
            "clear_addition_count_per_parameter",
            "clear_addition_count_per_tag",
            "clear_addition_count_per_tag_per_parameter",
            "encrypted_addition_count",
            "encrypted_addition_count_per_parameter",
            "encrypted_addition_count_per_tag",
            "encrypted_addition_count_per_tag_per_parameter",
            "clear_multiplication_count",
            "clear_multiplication_count_per_parameter",
            "clear_multiplication_count_per_tag",
            "clear_multiplication_count_per_tag_per_parameter",
            "encrypted_negation_count",
            "encrypted_negation_count_per_parameter",
            "encrypted_negation_count_per_tag",
            "encrypted_negation_count_per_tag_per_parameter",
        ]
        return {attribute: getattr(self, attribute) for attribute in attributes}


class FheModule:
    """
    Fhe module class, to combine computation graphs, mlir, runtime objects into a single object.
    """

    configuration: Configuration
    graphs: Dict[str, Graph]
    mlir_module: MlirModule
    compilation_context: CompilationContext
    runtime: Union[ExecutionRt, SimulationRt]

    def __init__(
        self,
        graphs: Dict[str, Graph],
        mlir: MlirModule,
        compilation_context: CompilationContext,
        configuration: Optional[Configuration] = None,
    ):
        assert configuration and (configuration.fhe_simulation or configuration.fhe_execution)

        self.configuration = configuration if configuration is not None else Configuration()
        self.graphs = graphs
        self.mlir_module = mlir
        self.compilation_context = compilation_context

        if self.configuration.fhe_simulation:
            server = Server.create(
                self.mlir_module,
                self.configuration,
                is_simulated=True,
                compilation_context=self.compilation_context,
            )
            self.runtime = SimulationRt(server)
        else:
            server = Server.create(
                self.mlir_module, self.configuration, compilation_context=self.compilation_context
            )

            keyset_cache_directory = None
            if self.configuration.use_insecure_key_cache:
                assert_that(self.configuration.enable_unsafe_features)
                assert_that(self.configuration.insecure_key_cache_location is not None)
                keyset_cache_directory = self.configuration.insecure_key_cache_location

            client = Client(server.client_specs, keyset_cache_directory)
            self.runtime = ExecutionRt(client, server)

    @property
    def mlir(self) -> str:
        """Textual representation of the MLIR module.

        Returns:
            str: textual representation of the MLIR module
        """
        return str(self.mlir_module).strip()

    @property
    def keys(self) -> Optional[Keys]:
        """
        Get the keys of the module.
        """
        if isinstance(self.runtime, ExecutionRt):
            return self.runtime.client.keys
        return None

    @keys.setter
    def keys(self, new_keys: Keys):
        """
        Set the keys of the module.
        """
        if isinstance(self.runtime, ExecutionRt):
            self.runtime.client.keys = new_keys

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
        if isinstance(self.runtime, ExecutionRt):
            self.runtime.client.keygen(force, seed, encryption_seed)

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """
        self.runtime.server.cleanup()

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the module.
        """
        return self.runtime.server.size_of_secret_keys  # pragma: no cover

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the module.
        """
        return self.runtime.server.size_of_bootstrap_keys  # pragma: no cover

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the module.
        """
        return self.runtime.server.size_of_keyswitch_keys  # pragma: no cover

    @property
    def p_error(self) -> int:
        """
        Get probability of error for each simple TLU (on a scalar).
        """
        return self.runtime.server.p_error  # pragma: no cover

    @property
    def global_p_error(self) -> int:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """
        return self.runtime.server.global_p_error  # pragma: no cover

    @property
    def complexity(self) -> float:
        """
        Get complexity of the module.
        """
        return self.runtime.server.complexity  # pragma: no cover

    @property
    def statistics(self) -> Dict:
        """
        Get all statistics of the module.
        """
        attributes = [
            "size_of_secret_keys",
            "size_of_bootstrap_keys",
            "size_of_keyswitch_keys",
            "p_error",
            "global_p_error",
            "complexity",
        ]
        statistics = {attribute: getattr(self, attribute) for attribute in attributes}
        statistics["functions"] = {
            name: function.statistics for (name, function) in self.functions().items()
        }
        return statistics

    def functions(self) -> Dict[str, FheFunction]:
        """
        Return a dictionnary containing all the functions of the module.
        """
        return {
            name: FheFunction(name, self.runtime, self.graphs[name]) for name in self.graphs.keys()
        }

    def __getattr__(self, item):
        if item not in list(self.graphs.keys()):
            error = f"No attribute {item}"
            raise AttributeError(error)
        return FheFunction(item, self.runtime, self.graphs[item])
