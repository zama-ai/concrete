"""
Declaration of `FheModule` classes.
"""

# pylint: disable=import-error,no-member,no-name-in-module

import asyncio
from collections.abc import Awaitable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Any, NamedTuple, Optional, Union

import numpy as np
from concrete.compiler import CompilationContext, LweSecretKey, Parameter
from mlir.ir import Module as MlirModule

from ..internal.utils import assert_that
from ..representation import Graph
from ..tfhers.specs import TFHERSClientSpecs
from .client import Client
from .composition import CompositionRule
from .configuration import Configuration
from .keys import Keys
from .server import Server
from .utils import Lazy
from .value import Value

# pylint: enable=import-error,no-member,no-name-in-module


class ExecutionRt:
    """
    Runtime object class for execution.
    """

    client: Client
    server: Server
    auto_schedule_run: bool
    fhe_executor_pool: ThreadPoolExecutor
    fhe_waiter_loop: asyncio.BaseEventLoop
    fhe_waiter_thread: Thread  # daemon thread

    def __init__(self, client, server, auto_schedule_run):
        self.client = client
        self.server = server
        self.auto_schedule_run = auto_schedule_run
        if auto_schedule_run:
            self.fhe_executor_pool = ThreadPoolExecutor()
            self.fhe_waiter_loop = asyncio.new_event_loop()

            def loop_thread():
                asyncio.set_event_loop(self.fhe_waiter_loop)
                self.fhe_waiter_loop.run_forever()

            self.fhe_waiter_thread = Thread(target=loop_thread, args=(), daemon=True)
            self.fhe_waiter_thread.start()
        else:
            self.fhe_executor_pool = None
            self.fhe_waiter_loop = None
            self.fhe_waiter_thread = None

    def __del__(self):
        if self.fhe_waiter_loop:
            self.fhe_waiter_loop.stop()  # pragma: no cover


class SimulationRt(NamedTuple):
    """
    Runtime object class for simulation.
    """

    client: Client
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
        tuple[Union[np.bool_, np.integer, np.floating, np.ndarray], ...],
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

    def _simulate_encrypt(
        self,
        *args: Optional[Union[int, np.ndarray, list]],
    ) -> Optional[Union[Value, tuple[Optional[Value], ...]]]:

        return self.simulation_runtime.val.client.simulate_encrypt(*args, function_name=self.name)

    def _simulate_run(
        self,
        *args: Optional[Union[Value, tuple[Optional[Value], ...]]],
    ) -> Union[Value, tuple[Value, ...]]:

        return self.simulation_runtime.val.server.run(*args, function_name=self.name)

    def _simulate_decrypt(
        self,
        *results: Union[Value, tuple[Value, ...]],
    ) -> Optional[Union[int, np.ndarray, tuple[Optional[Union[int, np.ndarray]], ...]]]:

        return self.simulation_runtime.val.client.simulate_decrypt(
            *results, function_name=self.name
        )

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

        return self._simulate_decrypt(self._simulate_run(self._simulate_encrypt(*args)))

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

        if self.configuration.simulate_encrypt_run_decrypt:
            return tuple(args) if len(args) > 1 else args[0]  # type: ignore
        return self.execution_runtime.val.client.encrypt(*args, function_name=self.name)

    def run_sync(
        self,
        *args: Optional[Union[Value, tuple[Optional[Value], ...]]],
    ) -> Any:
        """
        Evaluate the function synchronuously.

        Args:
            *args (Value):
                argument(s) for evaluation

        Returns:
            Union[Value, Tuple[Value, ...]]:
                result(s) of evaluation
        """

        return self._run(True, *args)

    def run_async(
        self, *args: Optional[Union[Value, tuple[Optional[Value], ...]]]
    ) -> Union[Value, tuple[Value, ...], Awaitable[Union[Value, tuple[Value, ...]]]]:
        """
        Evaluate the function asynchronuously.

        Args:
            *args (Value):
                argument(s) for evaluation

        Returns:
            Union[Awaitable[Value], Awaitable[Tuple[Value, ...]]]:
                result(s) a future of the evaluation
        """
        if (
            isinstance(self.execution_runtime.val, ExecutionRt)
            and not self.execution_runtime.val.fhe_executor_pool
        ):
            client = self.execution_runtime.val.client
            server = self.execution_runtime.val.server
            self.execution_runtime = Lazy(lambda: ExecutionRt(client, server, True))
            self.execution_runtime.val.auto_schedule_run = False

        return self._run(False, *args)

    def run(
        self,
        *args: Optional[Union[Value, tuple[Optional[Value], ...]]],
    ) -> Union[Value, tuple[Value, ...], Awaitable[Union[Value, tuple[Value, ...]]]]:
        """
        Evaluate the function.

        Args:
            *args (Value):
                argument(s) for evaluation

        Returns:
            Union[Value, Tuple[Value, ...], Awaitable[Union[Value, Tuple[Value, ...]]]]:
                result(s) of evaluation or future of result(s) of evaluation if configured with
                async_run=True
        """
        if isinstance(self.execution_runtime.val, ExecutionRt):
            auto_schedule_run = self.execution_runtime.val.auto_schedule_run
        else:
            auto_schedule_run = False  # pragma: no cover
        return self._run(not auto_schedule_run, *args)

    def _run(
        self,
        sync: bool,
        *args: Optional[Union[Value, tuple[Optional[Value], ...]]],
    ) -> Union[Value, tuple[Value, ...], Awaitable[Union[Value, tuple[Value, ...]]]]:
        """
        Evaluate the function.

        Args:
            *args (Value):
                argument(s) for evaluation

        Returns:
            Union[Value, Tuple[Value, ...], Awaitable[Union[Value, Tuple[Value, ...]]]]:
                result(s) of evaluation if sync=True else future of result(s) of evaluation
        """
        if self.configuration.simulate_encrypt_run_decrypt:
            return self._simulate_decrypt(self._simulate_run(*args))  # type: ignore

        assert isinstance(self.execution_runtime.val, ExecutionRt)

        fhe_work = lambda *args: self.execution_runtime.val.server.run(
            *args,
            evaluation_keys=self.execution_runtime.val.client.evaluation_keys,
            function_name=self.name,
        )

        def args_ready(args):
            return [arg.result() if isinstance(arg, Future) else arg for arg in args]

        if sync:
            return fhe_work(*args_ready(args))

        all_args_done = all(not isinstance(arg, Future) or arg.done() for arg in args)

        fhe_work_future = lambda *args: self.execution_runtime.val.fhe_executor_pool.submit(
            fhe_work, *args
        )
        if all_args_done:
            return fhe_work_future(*args_ready(args))  # type: ignore

        # waiting args to be ready with async coroutines
        # it only required one thread to run unlimited waits vs unlimited sync threads
        async def wait_async(arg):
            if not isinstance(arg, Future):
                return arg  # pragma: no cover
            if arg.done():
                return arg.result()  # pragma: no cover
            return await asyncio.wrap_future(arg, loop=self.execution_runtime.val.fhe_waiter_loop)

        async def args_ready_and_submit(*args):
            args = [await wait_async(arg) for arg in args]
            return await wait_async(fhe_work_future(*args))

        run_async = args_ready_and_submit(*args)
        return asyncio.run_coroutine_threadsafe(
            run_async, self.execution_runtime.val.fhe_waiter_loop
        )  # type: ignore

    def decrypt(
        self, *results: Union[Value, tuple[Value, ...], Awaitable[Union[Value, tuple[Value, ...]]]]
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

        if self.configuration.simulate_encrypt_run_decrypt:
            return tuple(results) if len(results) > 1 else results[0]  # type: ignore

        assert isinstance(self.execution_runtime.val, ExecutionRt)
        results = [res.result() if isinstance(res, Future) else res for res in results]
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
    def programmable_bootstrap_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of programmable bootstraps per bit width in the function.
        """
        return self.execution_runtime.val.server.programmable_bootstrap_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of programmable bootstraps per tag in the function.
        """
        return self.execution_runtime.val.server.programmable_bootstrap_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def programmable_bootstrap_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[int, int]]:
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
    def key_switch_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of key switches per parameter in the function.
        """
        return self.execution_runtime.val.server.key_switch_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def key_switch_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of key switches per tag in the function.
        """
        return self.execution_runtime.val.server.key_switch_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def key_switch_count_per_tag_per_parameter(self) -> dict[str, dict[Parameter, int]]:
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
    def packing_key_switch_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of packing key switches per parameter in the function.
        """
        return self.execution_runtime.val.server.packing_key_switch_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of packing key switches per tag in the function.
        """
        return self.execution_runtime.val.server.packing_key_switch_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def packing_key_switch_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
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
    def clear_addition_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of clear additions per parameter in the function.
        """
        return self.execution_runtime.val.server.clear_addition_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def clear_addition_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of clear additions per tag in the function.
        """
        return self.execution_runtime.val.server.clear_addition_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def clear_addition_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
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
    def encrypted_addition_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of encrypted additions per parameter in the function.
        """
        return self.execution_runtime.val.server.encrypted_addition_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of encrypted additions per tag in the function.
        """
        return self.execution_runtime.val.server.encrypted_addition_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_addition_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
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
    def clear_multiplication_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of clear multiplications per parameter in the function.
        """
        return self.execution_runtime.val.server.clear_multiplication_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of clear multiplications per tag in the function.
        """
        return self.execution_runtime.val.server.clear_multiplication_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def clear_multiplication_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
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
    def encrypted_negation_count_per_parameter(self) -> dict[Parameter, int]:
        """
        Get the number of encrypted negations per parameter in the function.
        """
        return self.execution_runtime.val.server.encrypted_negation_count_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag(self) -> dict[str, int]:
        """
        Get the number of encrypted negations per tag in the function.
        """
        return self.execution_runtime.val.server.encrypted_negation_count_per_tag(
            self.name
        )  # pragma: no cover

    @property
    def encrypted_negation_count_per_tag_per_parameter(
        self,
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of encrypted negations per tag per parameter in the function.
        """
        return self.execution_runtime.val.server.encrypted_negation_count_per_tag_per_parameter(
            self.name
        )  # pragma: no cover

    @property
    def statistics(self) -> dict:
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
    graphs: dict[str, Graph]
    mlir_module: MlirModule
    compilation_context: CompilationContext
    execution_runtime: Lazy[ExecutionRt]
    simulation_runtime: Lazy[SimulationRt]

    def __init__(
        self,
        graphs: dict[str, Graph],
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

        tfhers_specs = TFHERSClientSpecs.from_graphs(graphs)

        def init_simulation():
            simulation_server = Server.create(
                self.mlir_module,
                self.configuration.fork(fhe_simulation=True),
                is_simulated=True,
                compilation_context=self.compilation_context,
                tfhers_specs=tfhers_specs,
            )
            simulation_client = Client(simulation_server.client_specs, is_simulated=True)
            return SimulationRt(simulation_client, simulation_server)

        self.simulation_runtime = Lazy(init_simulation)
        if configuration.fhe_simulation:
            self.simulation_runtime.init()

        def init_execution():
            execution_server = Server.create(
                self.mlir_module,
                self.configuration.fork(fhe_simulation=False),
                compilation_context=self.compilation_context,
                composition_rules=composition_rules,
                is_simulated=False,
                tfhers_specs=tfhers_specs,
            )
            keyset_cache_directory = None
            if self.configuration.use_insecure_key_cache:
                assert_that(self.configuration.enable_unsafe_features)
                assert_that(self.configuration.insecure_key_cache_location is not None)
                keyset_cache_directory = self.configuration.insecure_key_cache_location
            execution_client = Client(
                execution_server.client_specs, keyset_cache_directory, is_simulated=False
            )
            return ExecutionRt(
                execution_client, execution_server, self.configuration.auto_schedule_run
            )

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
    def keys(self) -> Optional[Keys]:
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
    def p_error(self) -> float:
        """
        Get probability of error for each simple TLU (on a scalar).
        """
        return self.execution_runtime.val.server.p_error  # pragma: no cover

    @property
    def global_p_error(self) -> float:
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
    def statistics(self) -> dict:
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

    def functions(self) -> dict[str, FheFunction]:
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

    def __getattr__(self, item) -> FheFunction:
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
