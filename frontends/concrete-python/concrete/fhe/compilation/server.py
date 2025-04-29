"""
Declaration of `Server` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

import json
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

# mypy: disable-error-code=attr-defined
import concrete.compiler
import jsonpickle
import numpy as np
from concrete.compiler import (
    Backend,
    ClientProgram,
    CompilationContext,
    CompilationOptions,
    Compiler,
    KeyType,
    Library,
    MoreCircuitCompilationFeedback,
    OptimizerMultiParameterStrategy,
    OptimizerStrategy,
    Parameter,
    PrimitiveOperation,
    ProgramInfo,
    ServerProgram,
)
from concrete.compiler import Value as Value_
from concrete.compiler import lookup_runtime_lib, set_compiler_logging, set_llvm_debug_flag
from mlir.ir import Module as MlirModule

from ..tfhers.specs import TFHERSClientSpecs
from .composition import CompositionClause, CompositionRule
from .configuration import (
    DEFAULT_GLOBAL_P_ERROR,
    DEFAULT_P_ERROR,
    Configuration,
    MultiParameterStrategy,
    ParameterSelectionStrategy,
)
from .evaluation_keys import EvaluationKeys
from .specs import ClientSpecs
from .utils import Lazy, friendly_type_format
from .value import Value

# pylint: enable=import-error,no-member,no-name-in-module


class Server:
    """
    Server class, which can be used to perform homomorphic computation.
    """

    is_simulated: bool
    _library: Library
    _mlir: Optional[str]
    _configuration: Optional[Configuration]
    _composition_rules: Optional[list[CompositionRule]]
    _tfhers_specs: Optional[TFHERSClientSpecs]

    def __init__(
        self,
        library: Library,
        is_simulated: bool,
        composition_rules: Optional[list[CompositionRule]],
        tfhers_specs: Optional[TFHERSClientSpecs] = None,
    ):
        self.is_simulated = is_simulated
        self._library = library
        self._mlir = None
        self._composition_rules = composition_rules
        self._tfhers_specs = tfhers_specs

    @property
    def client_specs(self) -> ClientSpecs:
        """
        Return the associated client specs.
        """
        return ClientSpecs(self._library.get_program_info(), tfhers_specs=self._tfhers_specs)

    @staticmethod
    def create(
        mlir: Union[str, MlirModule],
        configuration: Configuration,
        is_simulated: bool = False,
        compilation_context: Optional[CompilationContext] = None,
        composition_rules: Optional[Iterable[CompositionRule]] = None,
        tfhers_specs: Optional[TFHERSClientSpecs] = None,
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

            composition_rules (Iterable[Tuple[str, int, str, int]]):
                composition rules to be applied when compiling

            tfhers_specs (Optional[TFHERSClientSpecs]):
                TFHE-rs client specs
        """

        backend = Backend.GPU if configuration.use_gpu else Backend.CPU
        options = CompilationOptions(backend)
        options.simulation(is_simulated)
        options.set_loop_parallelize(configuration.loop_parallelize)
        options.set_dataflow_parallelize(configuration.dataflow_parallelize)
        options.set_auto_parallelize(configuration.auto_parallelize)
        options.set_compress_evaluation_keys(configuration.compress_evaluation_keys)
        options.set_compress_input_ciphertexts(configuration.compress_input_ciphertexts)
        options.set_enable_overflow_detection_in_simulation(
            configuration.detect_overflow_in_simulation
        )
        options.set_composable(configuration.composable)
        composition_rules = list(composition_rules) if composition_rules else []
        for rule in composition_rules:
            options.add_composition(rule.from_.func, rule.from_.pos, rule.to.func, rule.to.pos)

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

        multi_parameter_strategy = configuration.multi_parameter_strategy
        converter = {
            MultiParameterStrategy.PRECISION: OptimizerMultiParameterStrategy.PRECISION,
            MultiParameterStrategy.PRECISION_AND_NORM2: (
                OptimizerMultiParameterStrategy.PRECISION_AND_NORM2
            ),
        }
        options.set_optimizer_multi_parameter_strategy(converter[multi_parameter_strategy])

        options.set_enable_tlu_fusing(configuration.enable_tlu_fusing)
        options.set_print_tlu_fusing(configuration.print_tlu_fusing)
        if configuration.keyset_restriction:
            options.set_keyset_restriction(configuration.keyset_restriction)

        if configuration.range_restriction:
            options.set_range_restriction(configuration.range_restriction)

        options.set_security_level(configuration.security_level)

        try:
            if configuration.compiler_debug_mode:  # pragma: no cover
                set_llvm_debug_flag(True)
            if configuration.compiler_verbose_mode:  # pragma: no cover
                set_compiler_logging(True)

            output_dir = tempfile.mkdtemp()
            output_dir_path = Path(output_dir)

            compiler = Compiler(
                str(output_dir_path),
                lookup_runtime_lib(),
                generate_shared_lib=True,
                generate_program_info=True,
                generate_compilation_feedback=True,
            )
            if isinstance(mlir, str):
                library = compiler.compile(mlir, options)
            else:  # MlirModule
                assert (
                    compilation_context is not None
                ), "must provide compilation context when compiling MlirModule"
                library = compiler.compile(
                    mlir._CAPIPtr, options, compilation_context  # pylint: disable=protected-access
                )
        finally:
            set_llvm_debug_flag(False)
            set_compiler_logging(False)

        composition_rules = composition_rules if composition_rules else None

        result = Server(
            library=library,
            is_simulated=is_simulated,
            composition_rules=composition_rules,
            tfhers_specs=tfhers_specs,
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
                    f.write(jsonpickle.dumps(self._configuration.__dict__))

                with open(Path(tmp) / "composition_rules.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(self._composition_rules))

                with open(Path(tmp) / "client.specs.json", "wb") as f:
                    f.write(self.client_specs.serialize())

                shutil.make_archive(path, "zip", tmp)

            return

        # Note that the shared library, program info and more are already in the output directory.
        # We just add a few things related to concrete-python here.
        with open(Path(self._library.get_output_dir_path()) / "client.specs.json", "wb") as f:
            f.write(self.client_specs.serialize())

        with open(
            Path(self._library.get_output_dir_path()) / "is_simulated", "w", encoding="utf-8"
        ) as f:
            f.write("1" if self.is_simulated else "0")

        with open(
            Path(self._library.get_output_dir_path()) / "composition_rules.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(self._composition_rules))

        shutil.make_archive(path, "zip", self._library.get_output_dir_path())

    @staticmethod
    def load(path: Union[str, Path], **kwargs) -> "Server":
        """
        Load the server from the given path in zip format.

        Args:
            path (Union[str, Path]):
                path to load the server from

            kwargs (Dict[str, Any]):
                configuration options to overwrite when loading a server saved with `via_mlir`
                if server isn't loaded via mlir, kwargs are ignored

        Returns:
            Server:
                server loaded from the filesystem
        """

        # pylint: disable=consider-using-with
        output_dir = tempfile.mkdtemp()
        output_dir_path = Path(output_dir)
        # pylint: enable=consider-using-with

        shutil.unpack_archive(path, str(output_dir_path), "zip")

        with open(output_dir_path / "is_simulated", "r", encoding="utf-8") as f:
            is_simulated = f.read() == "1"

        composition_rules = None
        if (output_dir_path / "composition_rules.json").exists():
            with open(output_dir_path / "composition_rules.json", "r", encoding="utf-8") as f:
                composition_rules = json.loads(f.read())
                composition_rules = (
                    [
                        CompositionRule(
                            CompositionClause(rule[0][0], rule[0][1]),
                            CompositionClause(rule[1][0], rule[1][1]),
                        )
                        for rule in composition_rules
                    ]
                    if composition_rules
                    else None
                )

        tfhers_specs = None
        if (output_dir_path / "client.specs.json").exists():
            with open(output_dir_path / "client.specs.json", "rb") as f:
                tfhers_specs = ClientSpecs.deserialize(f.read()).tfhers_specs

        if (output_dir_path / "circuit.mlir").exists():
            with open(output_dir_path / "circuit.mlir", "r", encoding="utf-8") as f:
                mlir = f.read()

            with open(output_dir_path / "configuration.json", "r", encoding="utf-8") as f:
                configuration = Configuration().fork(**jsonpickle.loads(f.read())).fork(**kwargs)

            return Server.create(
                mlir,
                configuration,
                is_simulated,
                composition_rules=composition_rules,
                tfhers_specs=tfhers_specs,
            )

        library = Library(str(output_dir_path))

        return Server(
            library,
            is_simulated,
            composition_rules,
            tfhers_specs=tfhers_specs,
        )

    def run(
        self,
        *args: Optional[Union[Value, tuple[Optional[Value], ...]]],
        evaluation_keys: Optional[EvaluationKeys] = None,
        function_name: Optional[str] = None,
    ) -> Union[Value, tuple[Value, ...]]:
        """
        Evaluate.

        Args:
            *args (Optional[Union[Value, Tuple[Optional[Value], ...]]]):
                argument(s) for evaluation

            evaluation_keys (Optional[EvaluationKeys], default = None):
                evaluation keys required for fhe execution

            function_name (str):
                The name of the function to run

        Returns:
            Union[Value, Tuple[Value, ...]]:
                result(s) of evaluation
        """

        if function_name is None:
            circuits = self.program_info.get_circuits()
            if len(circuits) == 1:
                function_name = circuits[0].get_name()
            else:  # pragma: no cover
                msg = "The server contains more than one functions. \
Provide a `function_name` keyword argument to disambiguate."
                raise TypeError(msg)

        if evaluation_keys is None and not self.is_simulated:
            message = "Expected evaluation keys to be provided when not in simulation mode"
            raise RuntimeError(message)

        flattened_args: list[Optional[Value]] = []
        for arg in args:
            if isinstance(arg, tuple):
                flattened_args.extend(arg)
            else:
                flattened_args.append(arg)

        if not self.is_simulated:
            for i, arg in enumerate(flattened_args):
                if arg is None:
                    message = f"Expected argument {i} to be an fhe.Value but it's None"
                    raise ValueError(message)

                if not isinstance(arg, Value):
                    if (
                        not self.client_specs.program_info.get_circuit(function_name)
                        .get_inputs()[i]
                        .get_type_info()
                        .is_plaintext()
                    ):
                        message = (
                            f"Expected argument {i} to be an fhe.Value "
                            f"but it's {friendly_type_format(type(arg))}"
                        )
                        raise ValueError(message)

        server_program = ServerProgram(self._library, self.is_simulated)
        server_circuit = server_program.get_server_circuit(function_name)

        def init_simulated_client_circuit():
            client_program = ClientProgram.create_simulated(self.client_specs.program_info)
            return client_program.get_client_circuit(function_name)

        simulated_client_circuit = Lazy(init_simulated_client_circuit)

        unwrapped_args = []
        for i, arg in enumerate(flattened_args):
            if isinstance(arg, Value):
                unwrapped_args.append(arg._inner)  # pylint: disable=protected-access
            elif isinstance(arg, list):
                unwrapped_args.append(
                    simulated_client_circuit.val.simulate_prepare_input(Value_(np.array(arg)), i)
                )
            else:
                unwrapped_args.append(
                    simulated_client_circuit.val.simulate_prepare_input(Value_(arg), i)
                )

        if self.is_simulated:
            result = server_circuit.simulate(unwrapped_args)
        else:
            assert evaluation_keys is not None
            result = server_circuit.call(unwrapped_args, evaluation_keys.server_keyset)

        result = [Value(r) for r in result]
        return tuple(result) if len(result) > 1 else result[0]

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """

        # if self._output_dir is not None:
        #     shutil.rmtree(Path(self._output_dir).resolve())

    @property
    def program_info(self) -> ProgramInfo:
        """
        The program info associated with the server.
        """
        return self._library.get_program_info()

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the compiled program.
        """
        return self._library.get_program_compilation_feedback().total_secret_keys_size

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the compiled program.
        """
        return self._library.get_program_compilation_feedback().total_bootstrap_keys_size

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the compiled program.
        """
        return self._library.get_program_compilation_feedback().total_keyswitch_keys_size

    @property
    def p_error(self) -> float:
        """
        Get the probability of error for each simple TLU (on a scalar).
        """
        return self._library.get_program_compilation_feedback().p_error

    @property
    def global_p_error(self) -> float:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """
        return self._library.get_program_compilation_feedback().global_p_error

    @property
    def complexity(self) -> float:
        """
        Get complexity of the compiled program.
        """
        return self._library.get_program_compilation_feedback().complexity

    def memory_usage_per_location(self, function: str) -> dict[str, Optional[int]]:
        """
        Get the memory usage of operations per location.
        """
        return (
            self._library.get_program_compilation_feedback()
            .get_circuit_feedback(function)
            .memory_usage_per_location
        )

    def size_of_inputs(self, function: str) -> int:
        """
        Get size of the inputs of the compiled program.
        """
        return (
            self._library.get_program_compilation_feedback()
            .get_circuit_feedback(function)
            .total_inputs_size
        )

    def size_of_outputs(self, function: str) -> int:
        """
        Get size of the outputs of the compiled program.
        """
        return (
            self._library.get_program_compilation_feedback()
            .get_circuit_feedback(function)
            .total_output_size
        )

    # Programmable Bootstrap Statistics

    def programmable_bootstrap_count(self, function: str) -> int:
        """
        Get the number of programmable bootstraps in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.PBS, PrimitiveOperation.WOP_PBS},
        )

    def programmable_bootstrap_count_per_parameter(self, function: str) -> dict[Parameter, int]:
        """
        Get the number of programmable bootstraps per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.PBS, PrimitiveOperation.WOP_PBS},
            key_types={KeyType.BOOTSTRAP},
            program_info=self.program_info,
        )

    def programmable_bootstrap_count_per_tag(self, function: str) -> dict[str, int]:
        """
        Get the number of programmable bootstraps per tag in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.PBS, PrimitiveOperation.WOP_PBS},
        )

    def programmable_bootstrap_count_per_tag_per_parameter(
        self, function: str
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of programmable bootstraps per tag per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.PBS, PrimitiveOperation.WOP_PBS},
            key_types={KeyType.BOOTSTRAP},
            program_info=self.program_info,
        )

    # Key Switch Statistics

    def key_switch_count(self, function: str) -> int:
        """
        Get the number of key switches in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.KEY_SWITCH, PrimitiveOperation.WOP_PBS},
        )

    def key_switch_count_per_parameter(self, function: str) -> dict[Parameter, int]:
        """
        Get the number of key switches per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.KEY_SWITCH, PrimitiveOperation.WOP_PBS},
            key_types={KeyType.KEY_SWITCH},
            program_info=self.program_info,
        )

    def key_switch_count_per_tag(self, function: str) -> dict[str, int]:
        """
        Get the number of key switches per tag in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.KEY_SWITCH, PrimitiveOperation.WOP_PBS},
        )

    def key_switch_count_per_tag_per_parameter(
        self, function: str
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of key switches per tag per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.KEY_SWITCH, PrimitiveOperation.WOP_PBS},
            key_types={KeyType.KEY_SWITCH},
            program_info=self.program_info,
        )

    # Packing Key Switch Statistics

    def packing_key_switch_count(self, function: str) -> int:
        """
        Get the number of packing key switches in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.WOP_PBS},
        )

    def packing_key_switch_count_per_parameter(self, function: str) -> dict[Parameter, int]:
        """
        Get the number of packing key switches per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.WOP_PBS},
            key_types={KeyType.PACKING_KEY_SWITCH},
            program_info=self.program_info,
        )

    def packing_key_switch_count_per_tag(self, function: str) -> dict[str, int]:
        """
        Get the number of packing key switches per tag in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.WOP_PBS},
        )

    def packing_key_switch_count_per_tag_per_parameter(
        self, function: str
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of packing key switches per tag per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.WOP_PBS},
            key_types={KeyType.PACKING_KEY_SWITCH},
            program_info=self.program_info,
        )

    # Clear Addition Statistics

    def clear_addition_count(self, function: str) -> int:
        """
        Get the number of clear additions in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.CLEAR_ADDITION},
        )

    def clear_addition_count_per_parameter(self, function: str) -> dict[Parameter, int]:
        """
        Get the number of clear additions per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.CLEAR_ADDITION},
            key_types={KeyType.SECRET},
            program_info=self.program_info,
        )

    def clear_addition_count_per_tag(self, function: str) -> dict[str, int]:
        """
        Get the number of clear additions per tag in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.CLEAR_ADDITION},
        )

    def clear_addition_count_per_tag_per_parameter(
        self, function: str
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of clear additions per tag per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.CLEAR_ADDITION},
            key_types={KeyType.SECRET},
            program_info=self.program_info,
        )

    # Encrypted Addition Statistics

    def encrypted_addition_count(self, function: str) -> int:
        """
        Get the number of encrypted additions in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.ENCRYPTED_ADDITION},
        )

    def encrypted_addition_count_per_parameter(self, function: str) -> dict[Parameter, int]:
        """
        Get the number of encrypted additions per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.ENCRYPTED_ADDITION},
            key_types={KeyType.SECRET},
            program_info=self.program_info,
        )

    def encrypted_addition_count_per_tag(self, function: str) -> dict[str, int]:
        """
        Get the number of encrypted additions per tag in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.ENCRYPTED_ADDITION},
        )

    def encrypted_addition_count_per_tag_per_parameter(
        self, function: str
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of encrypted additions per tag per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.ENCRYPTED_ADDITION},
            key_types={KeyType.SECRET},
            program_info=self.program_info,
        )

    # Clear Multiplication Statistics

    def clear_multiplication_count(self, function: str) -> int:
        """
        Get the number of clear multiplications in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.CLEAR_MULTIPLICATION},
        )

    def clear_multiplication_count_per_parameter(self, function: str) -> dict[Parameter, int]:
        """
        Get the number of clear multiplications per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.CLEAR_MULTIPLICATION},
            key_types={KeyType.SECRET},
            program_info=self.program_info,
        )

    def clear_multiplication_count_per_tag(self, function: str) -> dict[str, int]:
        """
        Get the number of clear multiplications per tag in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.CLEAR_MULTIPLICATION},
        )

    def clear_multiplication_count_per_tag_per_parameter(
        self, function: str
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of clear multiplications per tag per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.CLEAR_MULTIPLICATION},
            key_types={KeyType.SECRET},
            program_info=self.program_info,
        )

    # Encrypted Negation Statistics

    def encrypted_negation_count(self, function: str) -> int:
        """
        Get the number of encrypted negations in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.ENCRYPTED_NEGATION},
        )

    def encrypted_negation_count_per_parameter(self, function: str) -> dict[Parameter, int]:
        """
        Get the number of encrypted negations per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.ENCRYPTED_NEGATION},
            key_types={KeyType.SECRET},
            program_info=self.program_info,
        )

    def encrypted_negation_count_per_tag(self, function: str) -> dict[str, int]:
        """
        Get the number of encrypted negations per tag in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.ENCRYPTED_NEGATION},
        )

    def encrypted_negation_count_per_tag_per_parameter(
        self, function: str
    ) -> dict[str, dict[Parameter, int]]:
        """
        Get the number of encrypted negations per tag per parameter in the compiled program.
        """
        return MoreCircuitCompilationFeedback.count_per_tag_per_parameter(
            self._library.get_program_compilation_feedback().get_circuit_feedback(function),
            operations={PrimitiveOperation.ENCRYPTED_NEGATION},
            key_types={KeyType.SECRET},
            program_info=self.program_info,
        )
