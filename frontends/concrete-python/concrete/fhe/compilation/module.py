"""
Declaration of `FheModule` classes.
"""

# pylint: disable=import-error,no-member,no-name-in-module

from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from concrete.compiler import (
    CompilationContext,
    LweSecretKey,
    Parameter,
    SimulatedValueDecrypter,
    SimulatedValueExporter,
)
from mlir.ir import Module as MlirModule

from ..internal.utils import assert_that
from ..representation import Graph
from .client import Client
from .composition import CompositionRule
from .configuration import Configuration
from .keys import Keys
from .server import Server
from .utils import Lazy, validate_input_args
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

    execution_runtime: Lazy[ExecutionRt]
    simulation_runtime: Lazy[SimulationRt]
    graph: Graph
    name: str
    configuration: Configuration

    def __init__(
        self,
        name: str,
        execution_runtime: Lazy[ExecutionRt],
        simulation_runtime: Lazy[SimulationRt],
        graph: Graph,
        configuration: Configuration,
    ):
        self.name = name
        self.execution_runtime = execution_runtime
        self.simulation_runtime = simulation_runtime
        self.graph = graph
        self.configuration = configuration

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
        return self.graph.draw(  # pragma: no cover
            horizontal=horizontal, save_to=save_to, show=show
        )

    def __str__(self):
        return self.graph.format()

    def __repr__(self) -> str:
        return f"FheFunction(name={self.name})"

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

        ordered_validated_args = validate_input_args(
            self.simulation_runtime.val.server.client_specs,
            *args,
            function_name=self.name,
        )

        exporter = SimulatedValueExporter.new(
            self.simulation_runtime.val.server.client_specs.client_parameters, self.name
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

        results = self.simulation_runtime.val.server.run(*exported, function_name=self.name)
        if not isinstance(results, tuple):
            results = (results,)

        decrypter = SimulatedValueDecrypter.new(
            self.simulation_runtime.val.server.client_specs.client_parameters, self.name
        )
        decrypted = tuple(
            decrypter.decrypt(position, result.inner) for position, result in enumerate(results)
        )
        return decrypted if len(decrypted) != 1 else decrypted[0]

    def encrypt(
        self, *args: Optional[Union[int, np.ndarray, List]]
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

        if self.configuration.simulate_encrypt_run_decrypt:
            return args if len(args) != 1 else args[0]  # type: ignore

        return self.execution_runtime.val.client.encrypt(*args, function_name=self.name)

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

        if self.configuration.simulate_encrypt_run_decrypt:
            return self.simulate(*args)

        return self.execution_runtime.val.server.run(
            *args,
            evaluation_keys=self.execution_runtime.val.client.evaluation_keys,
            function_name=self.name,
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

        if self.configuration.simulate_encrypt_run_decrypt:
            return results if len(results) != 1 else results[0]  # type: ignore

        return self.execution_runtime.val.client.decrypt(*results, function_name=self.name)

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
        return self.execution_runtime.val.server.size_of_inputs(self.name)  # pragma: no cover

    @property
    def size_of_outputs(self) -> int:
        """
        Get size of the outputs of the function.
        """
        return self.execution_runtime.val.server.size_of_outputs(self.name)  # pragma: no cover

    # Programmable Bootstrap Statistics

    @property
    def programmable_bootstrap_count(self) -> int:
        """
        Get the number of programmable bootstraps in the function.
        """
        return self.execution_runtime.val.server.programmable_bootstrap_count(
            self.name
        )  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of programmable bootstraps per bit width in the function.
        """
        return self.execution_runtime.val.server.programmable_bootstrap_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of programmable bootstraps per tag in the function.
        """
        return self.execution_runtime.val.server.programmable_bootstrap_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag_per_parameter(
        self,
    ) -> Dict[str, Dict[int, int]]:
        """
        Get the number of programmable bootstraps per tag per bit width in the function.
        """
        server = self.execution_runtime.val.server
        return server.programmable_bootstrap_count_per_tag_per_parameter(self.name)
        # pragma: no cover

    # Key Switch Statistics

    @property
    def key_switch_count(self) -> int:
        """
        Get the number of key switches in the function.
        """
        return self.execution_runtime.val.server.key_switch_count(self.name)  # pragma: no cover

    @property
    def key_switch_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of key switches per parameter in the function.
        """
        return self.execution_runtime.val.server.key_switch_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def key_switch_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of key switches per tag in the function.
        """
        return self.execution_runtime.val.server.key_switch_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def key_switch_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of key switches per tag per parameter in the function.
        """
        return self.execution_runtime.val.server.key_switch_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Packing Key Switch Statistics

    @property
    def packing_key_switch_count(self) -> int:
        """
        Get the number of packing key switches in the function.
        """
        return self.execution_runtime.val.server.packing_key_switch_count(
            self.name
        )  # pragma: no cover

    @property
    def packing_key_switch_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of packing key switches per parameter in the function.
        """
        return self.execution_runtime.val.server.packing_key_switch_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of packing key switches per tag in the function.
        """
        return self.execution_runtime.val.server.packing_key_switch_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag_per_parameter(
        self,
    ) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of packing key switches per tag per parameter in the function.
        """
        return self.execution_runtime.val.server.packing_key_switch_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Clear Addition Statistics

    @property
    def clear_addition_count(self) -> int:
        """
        Get the number of clear additions in the function.
        """
        return self.execution_runtime.val.server.clear_addition_count(self.name)  # pragma: no cover

    @property
    def clear_addition_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of clear additions per parameter in the function.
        """
        return self.execution_runtime.val.server.clear_addition_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def clear_addition_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of clear additions per tag in the function.
        """
        return self.execution_runtime.val.server.clear_addition_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def clear_addition_count_per_tag_per_parameter(
        self,
    ) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of clear additions per tag per parameter in the function.
        """
        return self.execution_runtime.val.server.clear_addition_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Encrypted Addition Statistics

    @property
    def encrypted_addition_count(self) -> int:
        """
        Get the number of encrypted additions in the function.
        """
        return self.execution_runtime.val.server.encrypted_addition_count(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_addition_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of encrypted additions per parameter in the function.
        """
        return self.execution_runtime.val.server.encrypted_addition_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of encrypted additions per tag in the function.
        """
        return self.execution_runtime.val.server.encrypted_addition_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag_per_parameter(
        self,
    ) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of encrypted additions per tag per parameter in the function.
        """
        return self.execution_runtime.val.server.encrypted_addition_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    # Clear Multiplication Statistics

    @property
    def clear_multiplication_count(self) -> int:
        """
        Get the number of clear multiplications in the function.
        """
        return self.execution_runtime.val.server.clear_multiplication_count(
            self.name
        )  # pragma: no cover

    @property
    def clear_multiplication_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of clear multiplications per parameter in the function.
        """
        return self.execution_runtime.val.server.clear_multiplication_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of clear multiplications per tag in the function.
        """
        return self.execution_runtime.val.server.clear_multiplication_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag_per_parameter(
        self,
    ) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of clear multiplications per tag per parameter in the function.
        """
        server = self.execution_runtime.val.server
        return server.clear_multiplication_count_per_tag_per_parameter(self.name)
        # pragma: no cover

    # Encrypted Negation Statistics

    @property
    def encrypted_negation_count(self) -> int:
        """
        Get the number of encrypted negations in the function.
        """
        return self.execution_runtime.val.server.encrypted_negation_count(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_negation_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of encrypted negations per parameter in the function.
        """
        return self.execution_runtime.val.server.encrypted_negation_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of encrypted negations per tag in the function.
        """
        return self.execution_runtime.val.server.encrypted_negation_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag_per_parameter(
        self,
    ) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of encrypted negations per tag per parameter in the function.
        """
        return self.execution_runtime.val.server.encrypted_negation_count_per_tag_per_parameter(
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
    execution_runtime: Lazy[ExecutionRt]
    simulation_runtime: Lazy[SimulationRt]

    def __init__(
        self,
        graphs: Dict[str, Graph],
        mlir: MlirModule,
        compilation_context: CompilationContext,
        configuration: Optional[Configuration] = None,
        composition_rules: Optional[Iterable[CompositionRule]] = None,
    ):
        assert configuration
        self.configuration = configuration if configuration is not None else Configuration()
        self.graphs = graphs
        self.mlir_module = mlir
        self.compilation_context = compilation_context

        def init_simulation():
            simulation_server = Server.create(
                self.mlir_module,
                self.configuration.fork(fhe_simulation=True),
                is_simulated=True,
                compilation_context=self.compilation_context,
            )
            return SimulationRt(simulation_server)

        self.simulation_runtime = Lazy(init_simulation)
        if configuration.fhe_simulation:
            self.simulation_runtime.init()

        def init_execution():
            execution_server = Server.create(
                self.mlir_module,
                self.configuration.fork(fhe_simulation=False),
                compilation_context=self.compilation_context,
                composition_rules=composition_rules,
            )
            keyset_cache_directory = None
            if self.configuration.use_insecure_key_cache:
                assert_that(self.configuration.enable_unsafe_features)
                assert_that(self.configuration.insecure_key_cache_location is not None)
                keyset_cache_directory = self.configuration.insecure_key_cache_location
            execution_client = Client(execution_server.client_specs, keyset_cache_directory)
            return ExecutionRt(execution_client, execution_server)

        self.execution_runtime = Lazy(init_execution)
        if configuration.fhe_execution:
            self.execution_runtime.init()

    @property
    def mlir(self) -> str:
        """Textual representation of the MLIR module.

        Returns:
            str: textual representation of the MLIR module
        """
        return str(self.mlir_module).strip()

    @property
    def keys(self) -> Keys:
        """
        Get the keys of the module.
        """
        return self.execution_runtime.val.client.keys

    @keys.setter
    def keys(self, new_keys: Keys):
        """
        Set the keys of the module.
        """
        self.execution_runtime.val.client.keys = new_keys

    def keygen(
        self,
        force: bool = False,
        seed: Optional[int] = None,
        encryption_seed: Optional[int] = None,
        initial_keys: Optional[Dict[int, LweSecretKey]] = None,
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
        self.execution_runtime.val.client.keygen(force, seed, encryption_seed, initial_keys)

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """
        if self.execution_runtime.initialized:
            self.execution_runtime.val.server.cleanup()
        if self.simulation_runtime.initialized:
            self.simulation_runtime.val.server.cleanup()

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the module.
        """
        return self.execution_runtime.val.server.size_of_secret_keys  # pragma: no cover

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the module.
        """
        return self.execution_runtime.val.server.size_of_bootstrap_keys  # pragma: no cover

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the module.
        """

        return self.execution_runtime.val.server.size_of_keyswitch_keys  # pragma: no cover

    @property
    def p_error(self) -> int:
        """
        Get probability of error for each simple TLU (on a scalar).
        """
        return self.execution_runtime.val.server.p_error  # pragma: no cover

    @property
    def global_p_error(self) -> int:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """
        return self.execution_runtime.val.server.global_p_error  # pragma: no cover

    @property
    def complexity(self) -> float:
        """
        Get complexity of the module.
        """
        return self.execution_runtime.val.server.complexity  # pragma: no cover

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
            name: FheFunction(
                name,
                self.execution_runtime,
                self.simulation_runtime,
                self.graphs[name],
                self.configuration,
            )
            for name in self.graphs.keys()
        }

    @property
    def server(self) -> Server:
        """
        Get the execution server object tied to the module.
        """
        return self.execution_runtime.val.server

    @property
    def client(self) -> Client:
        """
        Returns the execution client object tied to the module.
        """
        return self.execution_runtime.val.client

    @property
    def simulator(self) -> Server:
        """
        Returns the simulation server object tied to the module.
        """
        return self.simulation_runtime.val.server

    @property
    def function_count(self) -> int:
        """
        Returns the number of functions in the module.
        """
        return len(self.graphs)

    def __getattr__(self, item):
        if item not in list(self.graphs.keys()):
            error = f"No attribute {item}"
            raise AttributeError(error)
        return FheFunction(
            item,
            self.execution_runtime,
            self.simulation_runtime,
            self.graphs[item],
            self.configuration,
        )
