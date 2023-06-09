"""
Declaration of `Server` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

# mypy: disable-error-code=attr-defined
import concrete.compiler
from concrete.compiler import (
    CompilationFeedback,
    CompilationOptions,
    EvaluationKeys,
    JITCompilationResult,
    JITLambda,
    JITSupport,
    LibraryCompilationResult,
    LibraryLambda,
    LibrarySupport,
    PublicArguments,
)
from mlir._mlir_libs._concretelang._compiler import OptimizerStrategy

from ..internal.utils import assert_that
from .configuration import (
    DEFAULT_GLOBAL_P_ERROR,
    DEFAULT_P_ERROR,
    Configuration,
    ParameterSelectionStrategy,
)
from .specs import ClientSpecs
from .value import Value

# pylint: enable=import-error,no-member,no-name-in-module


class Server:
    """
    Server class, which can be used to perform homomorphic computation.
    """

    client_specs: ClientSpecs

    _output_dir: Optional[tempfile.TemporaryDirectory]
    _support: Union[JITSupport, LibrarySupport]
    _compilation_result: Union[JITCompilationResult, LibraryCompilationResult]
    _compilation_feedback: CompilationFeedback
    _server_lambda: Union[JITLambda, LibraryLambda]

    _mlir: Optional[str]
    _configuration: Optional[Configuration]

    def __init__(
        self,
        client_specs: ClientSpecs,
        output_dir: Optional[tempfile.TemporaryDirectory],
        support: Union[JITSupport, LibrarySupport],
        compilation_result: Union[JITCompilationResult, LibraryCompilationResult],
        server_lambda: Union[JITLambda, LibraryLambda],
    ):
        self.client_specs = client_specs

        self._output_dir = output_dir
        self._support = support
        self._compilation_result = compilation_result
        self._compilation_feedback = self._support.load_compilation_feedback(compilation_result)
        self._server_lambda = server_lambda
        self._mlir = None

        assert_that(
            support.load_client_parameters(compilation_result).serialize()
            == client_specs.client_parameters.serialize()
        )

    @staticmethod
    def create(mlir: str, configuration: Configuration) -> "Server":
        """
        Create a server using MLIR and output sign information.

        Args:
            mlir (str):
                mlir to compile

            configuration (Optional[Configuration], default = None):
                configuration to use
        """

        options = CompilationOptions.new("main")

        options.set_loop_parallelize(configuration.loop_parallelize)
        options.set_dataflow_parallelize(configuration.dataflow_parallelize)
        options.set_auto_parallelize(configuration.auto_parallelize)

        if configuration.auto_parallelize or configuration.dataflow_parallelize:
            # pylint: disable=c-extension-no-member,no-member
            concrete.compiler.init_dfr()
            # pylint: enable=c-extension-no-member,no-member

        global_p_error_is_set = configuration.global_p_error is not None
        p_error_is_set = configuration.p_error is not None

        if global_p_error_is_set and p_error_is_set:  # pragma: no cover
            options.set_global_p_error(configuration.global_p_error)
            options.set_p_error(configuration.p_error)

        elif global_p_error_is_set:  # pragma: no cover
            options.set_global_p_error(configuration.global_p_error)
            options.set_p_error(1.0)

        elif p_error_is_set:  # pragma: no cover
            options.set_global_p_error(1.0)
            options.set_p_error(configuration.p_error)

        else:  # pragma: no cover
            if DEFAULT_GLOBAL_P_ERROR is not None:
                options.set_global_p_error(DEFAULT_GLOBAL_P_ERROR)
            else:
                options.set_global_p_error(1.0)

            if DEFAULT_P_ERROR is not None:
                options.set_p_error(DEFAULT_P_ERROR)
            else:
                options.set_p_error(1.0)

        show_optimizer = (
            configuration.show_optimizer
            if configuration.show_optimizer is not None
            else configuration.verbose
        )
        options.set_display_optimizer_choice(show_optimizer)

        parameter_selection_strategy = configuration.parameter_selection_strategy
        if parameter_selection_strategy == ParameterSelectionStrategy.V0:  # pragma: no cover
            options.set_optimizer_strategy(OptimizerStrategy.V0)
        elif parameter_selection_strategy == ParameterSelectionStrategy.MONO:  # pragma: no cover
            options.set_optimizer_strategy(OptimizerStrategy.DAG_MONO)
        elif parameter_selection_strategy == ParameterSelectionStrategy.MULTI:  # pragma: no cover
            options.set_optimizer_strategy(OptimizerStrategy.DAG_MULTI)

        if configuration.jit:
            output_dir = None

            support = JITSupport.new()
            compilation_result = support.compile(mlir, options)
            server_lambda = support.load_server_lambda(compilation_result)

        else:
            # pylint: disable=consider-using-with
            output_dir = tempfile.TemporaryDirectory()
            output_dir_path = Path(output_dir.name)
            # pylint: enable=consider-using-with

            support = LibrarySupport.new(
                str(output_dir_path), generateCppHeader=False, generateStaticLib=False
            )
            compilation_result = support.compile(mlir, options)
            server_lambda = support.load_server_lambda(compilation_result)

        client_parameters = support.load_client_parameters(compilation_result)
        client_specs = ClientSpecs(client_parameters)

        result = Server(client_specs, output_dir, support, compilation_result, server_lambda)

        # pylint: disable=protected-access
        result._mlir = mlir
        result._configuration = configuration
        # pylint: enable=protected-access

        return result

    def save(self, path: Union[str, Path], via_mlir: bool = False):
        """
        Save the server into the given path in zip format.

        Args:
            path (Union[str, Path]):
                path to save the server

            via_mlir (bool, default = False):
                export using the MLIR code of the program,
                this will make the export cross-platform
        """

        path = str(path)
        if path.endswith(".zip"):
            path = path[: len(path) - 4]

        if via_mlir:
            if self._mlir is None or self._configuration is None:
                message = "Loaded server objects cannot be saved again via MLIR"
                raise RuntimeError(message)

            with tempfile.TemporaryDirectory() as tmp:
                with open(Path(tmp) / "circuit.mlir", "w", encoding="utf-8") as f:
                    f.write(self._mlir)

                with open(Path(tmp) / "configuration.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(self._configuration.__dict__))

                shutil.make_archive(path, "zip", tmp)

            return

        if self._output_dir is None:
            message = "Just-in-Time compilation cannot be saved"
            raise RuntimeError(message)

        with open(Path(self._output_dir.name) / "client.specs.json", "wb") as f:
            f.write(self.client_specs.serialize())

        shutil.make_archive(path, "zip", self._output_dir.name)

    @staticmethod
    def load(path: Union[str, Path]) -> "Server":
        """
        Load the server from the given path in zip format.

        Args:
            path (Union[str, Path]):
                path to load the server from

        Returns:
            Server:
                server loaded from the filesystem
        """

        # pylint: disable=consider-using-with
        output_dir = tempfile.TemporaryDirectory()
        output_dir_path = Path(output_dir.name)
        # pylint: enable=consider-using-with

        shutil.unpack_archive(path, str(output_dir_path), "zip")

        if (output_dir_path / "circuit.mlir").exists():
            with open(output_dir_path / "circuit.mlir", "r", encoding="utf-8") as f:
                mlir = f.read()

            with open(output_dir_path / "configuration.json", "r", encoding="utf-8") as f:
                configuration = Configuration().fork(**json.load(f))

            return Server.create(mlir, configuration)

        with open(output_dir_path / "client.specs.json", "rb") as f:
            client_specs = ClientSpecs.deserialize(f.read())

        support = LibrarySupport.new(
            str(output_dir_path),
            generateCppHeader=False,
            generateStaticLib=False,
        )
        compilation_result = support.reload("main")
        server_lambda = support.load_server_lambda(compilation_result)

        return Server(client_specs, output_dir, support, compilation_result, server_lambda)

    def run(
        self,
        *args: Optional[Union[Value, Tuple[Optional[Value], ...]]],
        evaluation_keys: EvaluationKeys,
    ) -> Union[Value, Tuple[Value, ...]]:
        """
        Evaluate.

        Args:
            *args (Optional[Union[Value, Tuple[Optional[Value], ...]]]):
                argument(s) for evaluation

            evaluation_keys (EvaluationKeys):
                evaluation keys

        Returns:
            Union[Value, Tuple[Value, ...]]:
                result(s) of evaluation
        """

        flattened_args: List[Optional[Value]] = []
        for arg in args:
            if isinstance(arg, tuple):
                flattened_args.extend(arg)
            else:
                flattened_args.append(arg)

        buffers = []
        for i, arg in enumerate(flattened_args):
            if arg is None:
                message = f"Expected argument {i} to be an fhe.Value but it's None"
                raise ValueError(message)

            if not isinstance(arg, Value):
                message = f"Expected argument {i} to be an fhe.Value but it's {type(arg).__name__}"
                raise ValueError(message)

            buffers.append(arg.inner)

        public_args = PublicArguments.new(self.client_specs.client_parameters, buffers)
        public_result = self._support.server_call(self._server_lambda, public_args, evaluation_keys)

        result = tuple(Value(public_result.get_value(i)) for i in range(public_result.n_values()))
        return result if len(result) > 1 else result[0]

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """

        if self._output_dir is not None:
            self._output_dir.cleanup()

    @property
    def complexity(self) -> float:
        """
        Get complexity of the compiled program.
        """
        return self._compilation_feedback.complexity

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the compiled program.
        """
        return self._compilation_feedback.total_secret_keys_size

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the compiled program.
        """
        return self._compilation_feedback.total_bootstrap_keys_size

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the compiled program.
        """
        return self._compilation_feedback.total_keyswitch_keys_size

    @property
    def size_of_inputs(self) -> int:
        """
        Get size of the inputs of the compiled program.
        """
        return self._compilation_feedback.total_inputs_size

    @property
    def size_of_outputs(self) -> int:
        """
        Get size of the outputs of the compiled program.
        """
        return self._compilation_feedback.total_output_size

    @property
    def p_error(self) -> int:
        """
        Get the probability of error for each simple TLU (on a scalar).
        """
        return self._compilation_feedback.p_error

    @property
    def global_p_error(self) -> int:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """
        return self._compilation_feedback.global_p_error
