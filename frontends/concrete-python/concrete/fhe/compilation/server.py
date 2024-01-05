"""
Declaration of `Server` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# mypy: disable-error-code=attr-defined
import concrete.compiler
from concrete.compiler import (
    CompilationContext,
    CompilationFeedback,
    CompilationOptions,
    EvaluationKeys,
    LibraryCompilationResult,
    LibraryLambda,
    LibrarySupport,
    Parameter,
    PublicArguments,
    set_compiler_logging,
    set_llvm_debug_flag,
)
from mlir._mlir_libs._concretelang._compiler import (
    Backend,
    KeyType,
    OptimizerStrategy,
    PrimitiveOperation,
)
from mlir.ir import Module as MlirModule

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
    is_simulated: bool

    _output_dir: Optional[tempfile.TemporaryDirectory]
    _support: LibrarySupport
    _compilation_result: LibraryCompilationResult
    _compilation_feedback: CompilationFeedback
    _server_lambda: LibraryLambda

    _mlir: Optional[str]
    _configuration: Optional[Configuration]

    def __init__(
        self,
        client_specs: ClientSpecs,
        output_dir: Optional[tempfile.TemporaryDirectory],
        support: LibrarySupport,
        compilation_result: LibraryCompilationResult,
        server_lambda: LibraryLambda,
        is_simulated: bool,
    ):
        self.client_specs = client_specs
        self.is_simulated = is_simulated

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
    def create(
        mlir: Union[str, MlirModule],
        configuration: Configuration,
        is_simulated: bool = False,
        compilation_context: Optional[CompilationContext] = None,
    ) -> "Server":
        """
        Create a server using MLIR and output sign information.

        Args:
            mlir (MlirModule):
                mlir to compile

            is_simulated (bool, default = False):
                whether to compile in simulation mode or not

            configuration (Optional[Configuration]):
                configuration to use

            compilation_context (CompilationContext):
                context to use for the Compiler
        """

        backend = Backend.GPU if configuration.use_gpu else Backend.CPU
        options = CompilationOptions.new("main", backend)

        options.simulation(is_simulated)

        options.set_loop_parallelize(configuration.loop_parallelize)
        options.set_dataflow_parallelize(configuration.dataflow_parallelize)
        options.set_auto_parallelize(configuration.auto_parallelize)
        options.set_compress_evaluation_keys(configuration.compress_evaluation_keys)
        options.set_composable(configuration.composable)

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
        try:
            if configuration.compiler_debug_mode:  # pragma: no cover
                set_llvm_debug_flag(True)
            if configuration.compiler_verbose_mode:  # pragma: no cover
                set_compiler_logging(True)

            # pylint: disable=consider-using-with
            output_dir = tempfile.TemporaryDirectory()
            output_dir_path = Path(output_dir.name)
            # pylint: enable=consider-using-with

            support = LibrarySupport.new(
                str(output_dir_path), generateCppHeader=False, generateStaticLib=False
            )
            if isinstance(mlir, str):
                compilation_result = support.compile(mlir, options)
            else:  # MlirModule
                assert (
                    compilation_context is not None
                ), "must provide compilation context when compiling MlirModule"
                compilation_result = support.compile(mlir, options, compilation_context)
            server_lambda = support.load_server_lambda(compilation_result, is_simulated)
        finally:
            set_llvm_debug_flag(False)
            set_compiler_logging(False)

        client_parameters = support.load_client_parameters(compilation_result)
        client_specs = ClientSpecs(client_parameters)

        result = Server(
            client_specs,
            output_dir,
            support,
            compilation_result,
            server_lambda,
            is_simulated,
        )

        # pylint: disable=protected-access
        result._mlir = str(mlir).strip()
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

                with open(Path(tmp) / "is_simulated", "w", encoding="utf-8") as f:
                    f.write("1" if self.is_simulated else "0")

                with open(Path(tmp) / "configuration.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(self._configuration.__dict__))

                shutil.make_archive(path, "zip", tmp)

            return

        if self._output_dir is None:  # pragma: no cover
            message = "Output directory must be provided"
            raise RuntimeError(message)

        with open(Path(self._output_dir.name) / "client.specs.json", "wb") as f:
            f.write(self.client_specs.serialize())

        with open(Path(self._output_dir.name) / "is_simulated", "w", encoding="utf-8") as f:
            f.write("1" if self.is_simulated else "0")

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

        with open(output_dir_path / "is_simulated", "r", encoding="utf-8") as f:
            is_simulated = f.read() == "1"

        if (output_dir_path / "circuit.mlir").exists():
            with open(output_dir_path / "circuit.mlir", "r", encoding="utf-8") as f:
                mlir = f.read()

            with open(output_dir_path / "configuration.json", "r", encoding="utf-8") as f:
                configuration = Configuration().fork(**json.load(f))

            return Server.create(mlir, configuration, is_simulated)

        with open(output_dir_path / "client.specs.json", "rb") as f:
            client_specs = ClientSpecs.deserialize(f.read())

        support = LibrarySupport.new(
            str(output_dir_path),
            generateCppHeader=False,
            generateStaticLib=False,
        )
        compilation_result = support.reload("main")
        server_lambda = support.load_server_lambda(compilation_result, is_simulated)

        return Server(
            client_specs, output_dir, support, compilation_result, server_lambda, is_simulated
        )

    def run(
        self,
        *args: Optional[Union[Value, Tuple[Optional[Value], ...]]],
        evaluation_keys: Optional[EvaluationKeys] = None,
    ) -> Union[Value, Tuple[Value, ...]]:
        """
        Evaluate.

        Args:
            *args (Optional[Union[Value, Tuple[Optional[Value], ...]]]):
                argument(s) for evaluation

            evaluation_keys (Optional[EvaluationKeys], default = None):
                evaluation keys required for fhe execution

        Returns:
            Union[Value, Tuple[Value, ...]]:
                result(s) of evaluation
        """

        if evaluation_keys is None and not self.is_simulated:
            message = "Expected evaluation keys to be provided when not in simulation mode"
            raise RuntimeError(message)

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

        if self.is_simulated:
            public_result = self._support.simulate(self._server_lambda, public_args)
        else:
            public_result = self._support.server_call(
                self._server_lambda, public_args, evaluation_keys
            )

        result = tuple(Value(public_result.get_value(i)) for i in range(public_result.n_values()))
        return result if len(result) > 1 else result[0]

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """

        if self._output_dir is not None:
            self._output_dir.cleanup()

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

    @property
    def complexity(self) -> float:
        """
        Get complexity of the compiled program.
        """
        return self._compilation_feedback.complexity

    # Programmable Bootstrap Statistics

    @property
    def programmable_bootstrap_count(self) -> int:
        """
        Get the number of programmable bootstraps in the compiled program.
        """
        return self._compilation_feedback.count(
            operations={PrimitiveOperation.PBS, PrimitiveOperation.WOP_PBS},
        )

    @property
    def programmable_bootstrap_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of programmable bootstraps per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_parameter(
            operations={PrimitiveOperation.PBS, PrimitiveOperation.WOP_PBS},
            key_types={KeyType.BOOTSTRAP},
            client_parameters=self.client_specs.client_parameters,
        )

    @property
    def programmable_bootstrap_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of programmable bootstraps per tag in the compiled program.
        """
        return self._compilation_feedback.count_per_tag(
            operations={PrimitiveOperation.PBS, PrimitiveOperation.WOP_PBS},
        )

    @property
    def programmable_bootstrap_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of programmable bootstraps per tag per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_tag_per_parameter(
            operations={PrimitiveOperation.PBS, PrimitiveOperation.WOP_PBS},
            key_types={KeyType.BOOTSTRAP},
            client_parameters=self.client_specs.client_parameters,
        )

    # Key Switch Statistics

    @property
    def key_switch_count(self) -> int:
        """
        Get the number of key switches in the compiled program.
        """
        return self._compilation_feedback.count(
            operations={PrimitiveOperation.KEY_SWITCH, PrimitiveOperation.WOP_PBS},
        )

    @property
    def key_switch_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of key switches per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_parameter(
            operations={PrimitiveOperation.KEY_SWITCH, PrimitiveOperation.WOP_PBS},
            key_types={KeyType.KEY_SWITCH},
            client_parameters=self.client_specs.client_parameters,
        )

    @property
    def key_switch_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of key switches per tag in the compiled program.
        """
        return self._compilation_feedback.count_per_tag(
            operations={PrimitiveOperation.KEY_SWITCH, PrimitiveOperation.WOP_PBS},
        )

    @property
    def key_switch_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of key switches per tag per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_tag_per_parameter(
            operations={PrimitiveOperation.KEY_SWITCH, PrimitiveOperation.WOP_PBS},
            key_types={KeyType.KEY_SWITCH},
            client_parameters=self.client_specs.client_parameters,
        )

    # Packing Key Switch Statistics

    @property
    def packing_key_switch_count(self) -> int:
        """
        Get the number of packing key switches in the compiled program.
        """
        return self._compilation_feedback.count(operations={PrimitiveOperation.WOP_PBS})

    @property
    def packing_key_switch_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of packing key switches per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_parameter(
            operations={PrimitiveOperation.WOP_PBS},
            key_types={KeyType.PACKING_KEY_SWITCH},
            client_parameters=self.client_specs.client_parameters,
        )

    @property
    def packing_key_switch_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of packing key switches per tag in the compiled program.
        """
        return self._compilation_feedback.count_per_tag(operations={PrimitiveOperation.WOP_PBS})

    @property
    def packing_key_switch_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of packing key switches per tag per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_tag_per_parameter(
            operations={PrimitiveOperation.WOP_PBS},
            key_types={KeyType.PACKING_KEY_SWITCH},
            client_parameters=self.client_specs.client_parameters,
        )

    # Clear Addition Statistics

    @property
    def clear_addition_count(self) -> int:
        """
        Get the number of clear additions in the compiled program.
        """
        return self._compilation_feedback.count(operations={PrimitiveOperation.CLEAR_ADDITION})

    @property
    def clear_addition_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of clear additions per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_parameter(
            operations={PrimitiveOperation.CLEAR_ADDITION},
            key_types={KeyType.SECRET},
            client_parameters=self.client_specs.client_parameters,
        )

    @property
    def clear_addition_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of clear additions per tag in the compiled program.
        """
        return self._compilation_feedback.count_per_tag(
            operations={PrimitiveOperation.CLEAR_ADDITION},
        )

    @property
    def clear_addition_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of clear additions per tag per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_tag_per_parameter(
            operations={PrimitiveOperation.CLEAR_ADDITION},
            key_types={KeyType.SECRET},
            client_parameters=self.client_specs.client_parameters,
        )

    # Encrypted Addition Statistics

    @property
    def encrypted_addition_count(self) -> int:
        """
        Get the number of encrypted additions in the compiled program.
        """
        return self._compilation_feedback.count(operations={PrimitiveOperation.ENCRYPTED_ADDITION})

    @property
    def encrypted_addition_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of encrypted additions per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_parameter(
            operations={PrimitiveOperation.ENCRYPTED_ADDITION},
            key_types={KeyType.SECRET},
            client_parameters=self.client_specs.client_parameters,
        )

    @property
    def encrypted_addition_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of encrypted additions per tag in the compiled program.
        """
        return self._compilation_feedback.count_per_tag(
            operations={PrimitiveOperation.ENCRYPTED_ADDITION},
        )

    @property
    def encrypted_addition_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of encrypted additions per tag per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_tag_per_parameter(
            operations={PrimitiveOperation.ENCRYPTED_ADDITION},
            key_types={KeyType.SECRET},
            client_parameters=self.client_specs.client_parameters,
        )

    # Clear Multiplication Statistics

    @property
    def clear_multiplication_count(self) -> int:
        """
        Get the number of clear multiplications in the compiled program.
        """
        return self._compilation_feedback.count(
            operations={PrimitiveOperation.CLEAR_MULTIPLICATION},
        )

    @property
    def clear_multiplication_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of clear multiplications per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_parameter(
            operations={PrimitiveOperation.CLEAR_MULTIPLICATION},
            key_types={KeyType.SECRET},
            client_parameters=self.client_specs.client_parameters,
        )

    @property
    def clear_multiplication_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of clear multiplications per tag in the compiled program.
        """
        return self._compilation_feedback.count_per_tag(
            operations={PrimitiveOperation.CLEAR_MULTIPLICATION},
        )

    @property
    def clear_multiplication_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of clear multiplications per tag per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_tag_per_parameter(
            operations={PrimitiveOperation.CLEAR_MULTIPLICATION},
            key_types={KeyType.SECRET},
            client_parameters=self.client_specs.client_parameters,
        )

    # Encrypted Negation Statistics

    @property
    def encrypted_negation_count(self) -> int:
        """
        Get the number of encrypted negations in the compiled program.
        """
        return self._compilation_feedback.count(operations={PrimitiveOperation.ENCRYPTED_NEGATION})

    @property
    def encrypted_negation_count_per_parameter(self) -> Dict[Parameter, int]:
        """
        Get the number of encrypted negations per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_parameter(
            operations={PrimitiveOperation.ENCRYPTED_NEGATION},
            key_types={KeyType.SECRET},
            client_parameters=self.client_specs.client_parameters,
        )

    @property
    def encrypted_negation_count_per_tag(self) -> Dict[str, int]:
        """
        Get the number of encrypted negations per tag in the compiled program.
        """
        return self._compilation_feedback.count_per_tag(
            operations={PrimitiveOperation.ENCRYPTED_NEGATION},
        )

    @property
    def encrypted_negation_count_per_tag_per_parameter(self) -> Dict[str, Dict[Parameter, int]]:
        """
        Get the number of encrypted negations per tag per parameter in the compiled program.
        """
        return self._compilation_feedback.count_per_tag_per_parameter(
            operations={PrimitiveOperation.ENCRYPTED_NEGATION},
            key_types={KeyType.SECRET},
            client_parameters=self.client_specs.client_parameters,
        )

    # All Statistics

    @property
    def statistics(self) -> Dict:
        """
        Get all statistics of the compiled program.
        """
        attributes = [
            "size_of_secret_keys",
            "size_of_bootstrap_keys",
            "size_of_keyswitch_keys",
            "size_of_inputs",
            "size_of_outputs",
            "p_error",
            "global_p_error",
            "complexity",
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
