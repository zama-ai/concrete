"""
Declaration of `DebugArtifacts` class.
"""

import inspect
import platform
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

from ..representation import Graph
from .configuration import Configuration
from .utils import get_terminal_size

if TYPE_CHECKING:  # pragma: no cover
    from .module import ExecutionRt
    from .utils import Lazy

DEFAULT_OUTPUT_DIRECTORY: Path = Path(".artifacts")


class DebugManager:
    """
    A debug manager, allowing streamlined debugging.
    """

    configuration: Configuration
    begin_call: Callable

    def __init__(self, config: Configuration):
        self.configuration = config
        is_first = [True]

        def begin_call():
            if is_first[0]:
                print()
                is_first[0] = False

        self.begin_call = begin_call

    def debug_table(self, title: str, activate: bool = True):
        """
        Return a context manager that prints a table around what is printed inside the scope.
        """

        # pylint: disable=missing-class-docstring
        class DebugTableCm:
            def __init__(self, title):
                self.title = title
                self.columns = get_terminal_size()

            def __enter__(self):
                print(f"{self.title}")
                print("-" * self.columns)

            def __exit__(self, _exc_type, _exc_value, _exc_tb):
                print("-" * self.columns)
                print()

        class EmptyCm:
            def __enter__(self):
                pass

            def __exit__(self, _exc_type, _exc_value, _exc_tb):
                pass

        if activate:
            self.begin_call()
            return DebugTableCm(title)
        return EmptyCm()

    def show_graph(self) -> bool:
        """
        Tell if the configuration involves showing graph.
        """

        return (
            self.configuration.show_graph
            if self.configuration.show_graph is not None
            else self.configuration.verbose
        )

    def show_bit_width_constraints(self) -> bool:
        """
        Tell if the configuration involves showing bitwidth constraints.
        """

        return (
            self.configuration.show_bit_width_constraints
            if self.configuration.show_bit_width_constraints is not None
            else self.configuration.verbose
        )

    def show_bit_width_assignments(self) -> bool:
        """
        Tell if the configuration involves showing bitwidth assignments.
        """

        return (
            self.configuration.show_bit_width_assignments
            if self.configuration.show_bit_width_assignments is not None
            else self.configuration.verbose
        )

    def show_assigned_graph(self) -> bool:
        """
        Tell if the configuration involves showing assigned graph.
        """

        return (
            self.configuration.show_assigned_graph
            if self.configuration.show_assigned_graph is not None
            else self.configuration.verbose
        )

    def show_mlir(self) -> bool:
        """
        Tell if the configuration involves showing mlir.
        """

        return (
            self.configuration.show_mlir
            if self.configuration.show_mlir is not None
            else self.configuration.verbose
        )

    def show_optimizer(self) -> bool:
        """
        Tell if the configuration involves showing optimizer.
        """

        return (
            self.configuration.show_optimizer
            if self.configuration.show_optimizer is not None
            else self.configuration.verbose
        )

    def show_statistics(self) -> bool:
        """
        Tell if the configuration involves showing statistics.
        """

        return (
            self.configuration.show_statistics
            if self.configuration.show_statistics is not None
            else self.configuration.verbose
        )

    def debug_computation_graph(self, name, function_graph):
        """
        Print computation graph if configuration tells so.
        """

        if (
            self.show_graph()
            or self.show_bit_width_constraints()
            or self.show_bit_width_assignments()
            or self.show_assigned_graph()
            or self.show_mlir()
            or self.show_optimizer()
            or self.show_statistics()
        ):
            if self.show_graph():
                with self.debug_table(f"Computation Graph for {name}"):
                    print(function_graph.format())

    def debug_bit_width_constaints(self, name, function_graph):
        """
        Print bitwidth constraints if configuration tells so.
        """

        if self.show_bit_width_constraints():
            with self.debug_table(f"Bit-Width Constraints for {name}"):
                print(function_graph.format_bit_width_constraints())

    def debug_bit_width_assignments(self, name, function_graph):
        """
        Print bitwidth assignments if configuration tells so.
        """

        if self.show_bit_width_assignments():
            with self.debug_table(f"Bit-Width Assignments for {name}"):
                print(function_graph.format_bit_width_assignments())

    def debug_assigned_graph(self, name, function_graph):
        """
        Print assigned graphs if configuration tells so.
        """

        if self.show_assigned_graph():
            with self.debug_table(f"Bit-Width Assigned Computation Graph for {name}"):
                print(function_graph.format(show_assigned_bit_widths=True))

    def debug_mlir(self, mlir_str):
        """
        Print mlir if configuration tells so.
        """

        if self.show_mlir():
            with self.debug_table("MLIR"):
                print(mlir_str)

    def debug_statistics(self, module):
        """
        Print statistics if configuration tells so.
        """

        if self.show_statistics():

            def pretty(d, indent=0):  # pragma: no cover
                if indent > 0:
                    print("{")

                for key, value in d.items():
                    if isinstance(value, dict) and len(value) == 0:
                        continue
                    print("    " * indent + str(key) + ": ", end="")
                    if isinstance(value, dict):
                        pretty(value, indent + 1)
                    else:
                        print(value)
                if indent > 0:
                    print("    " * (indent - 1) + "}")

            with self.debug_table("Statistics"):
                pretty(module.statistics)


class FunctionDebugArtifacts:
    """
    An object containing debug artifacts for a certain function in an fhe module.
    """

    source_code: Optional[str]
    parameter_encryption_statuses: dict[str, str]
    textual_representations_of_graphs: dict[str, list[str]]
    final_graph: Optional[Graph]

    def __init__(self):
        self.source_code = None
        self.parameter_encryption_statuses = {}
        self.textual_representations_of_graphs = {}
        self.final_graph = None

    def add_source_code(self, function: Union[str, Callable]):
        """
        Add source code of the function being compiled.

        Args:
            function (Union[str, Callable]):
                either the source code of the function or the function itself
        """
        try:
            self.source_code = (
                function if isinstance(function, str) else inspect.getsource(function)
            )
        except OSError:  # pragma: no cover
            self.source_code = "unavailable"

    def add_parameter_encryption_status(self, name: str, encryption_status: str):
        """
        Add parameter encryption status of a parameter of the function being compiled.

        Args:
            name (str):
                name of the parameter

            encryption_status (str):
                encryption status of the parameter
        """
        self.parameter_encryption_statuses[name] = encryption_status

    def add_graph(self, name: str, graph: Graph):
        """
        Add a representation of the function being compiled.

        Args:
            name (str):
                name of the graph (e.g., initial, optimized, final)

            graph (Graph):
                a representation of the function being compiled
        """
        if name not in self.textual_representations_of_graphs:
            self.textual_representations_of_graphs[name] = []
        textual_representation = graph.format()
        self.textual_representations_of_graphs[name].append(textual_representation)
        self.final_graph = graph


class ModuleDebugArtifacts:
    """
    An object containing debug artifacts for an fhe module.
    """

    output_directory: Path
    mlir_to_compile: Optional[str]
    _execution_runtime: Optional["Lazy[ExecutionRt]"]
    functions: dict[str, FunctionDebugArtifacts]

    def __init__(
        self,
        function_names: Optional[list[str]] = None,
        output_directory: Union[str, Path] = DEFAULT_OUTPUT_DIRECTORY,
    ):
        self.output_directory = Path(output_directory)
        self.mlir_to_compile = None
        self._execution_runtime = None
        self.functions = (
            {name: FunctionDebugArtifacts() for name in function_names} if function_names else {}
        )

    def add_mlir_to_compile(self, mlir: str):
        """
        Add textual representation of the resulting MLIR.

        Args:
            mlir (str):
                textual representation of the resulting MLIR
        """
        self.mlir_to_compile = mlir

    def add_execution_runtime(self, execution_runtime: "Lazy[ExecutionRt]"):
        """
        Add the (lazy) execution runtime to get the client parameters if needed.

        Args:
            execution_runtime (Lazy[ExecutionRt]):
                The lazily initialized execution runtime.
        """

        self._execution_runtime = execution_runtime

    @property
    def client_parameters(self) -> Optional[bytes]:
        """
        The client parameters associated with the execution runtime.
        """

        return (
            self._execution_runtime.val.client.specs.program_info.serialize()
            if self._execution_runtime is not None
            else None
        )

    def export(self):
        """
        Export the collected information to `self.output_directory`.
        """
        # pylint: disable=too-many-branches

        output_directory = self.output_directory
        if output_directory.exists():
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True)

        with open(output_directory.joinpath("environment.txt"), "w", encoding="utf-8") as f:
            f.write(f"{platform.platform()} {platform.version()}\n")
            f.write(f"Python {platform.python_version()}\n")

        with open(output_directory.joinpath("requirements.txt"), "w", encoding="utf-8") as f:
            try:
                # example `pip list` output

                # Package                       Version
                # ----------------------------- ---------
                # alabaster                     0.7.12
                # appdirs                       1.4.4
                # ...                           ...
                # ...                           ...
                # wrapt                         1.12.1
                # zipp                          3.5.0

                # S603 `subprocess` call: check for execution of untrusted input
                # S607 Starting a process with a partial executable path
                pip_process = subprocess.run(  # noqa: S603
                    ["pip", "--disable-pip-version-check", "list"],  # noqa: S607
                    stdout=subprocess.PIPE,
                    check=True,
                )
                dependencies = iter(pip_process.stdout.decode("utf-8").split("\n"))

                # skip 'Package ... Version' line
                next(dependencies)

                # skip '------- ... -------' line
                next(dependencies)

            except Exception:  # pragma: no cover  # pylint: disable=broad-exception-caught
                dependencies = []  # pragma: no cover

            for dependency in dependencies:
                tokens = [token for token in dependency.split(" ") if token != ""]
                if len(tokens) == 0:
                    continue

                name = tokens[0]
                version = tokens[1]

                f.write(f"{name}=={version}\n")

        for function_name, function in self.functions.items():
            if function.source_code is not None:
                with open(
                    output_directory.joinpath(f"{function_name}.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(function.source_code)

            if len(function.parameter_encryption_statuses) > 0:
                with open(
                    output_directory.joinpath(f"{function_name}.parameters.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for name, parameter in function.parameter_encryption_statuses.items():
                        f.write(f"{name} :: {parameter}\n")

            identifier = 0

            textual_representations = function.textual_representations_of_graphs.items()
            for name, representations in textual_representations:
                for representation in representations:
                    identifier += 1
                    output_path = output_directory.joinpath(
                        f"{function_name}.{identifier}.{name}.graph.txt"
                    )
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(f"{representation}\n")

        if self.mlir_to_compile is not None:
            with open(output_directory.joinpath("mlir.txt"), "w", encoding="utf-8") as f:
                f.write(f"{self.mlir_to_compile}\n")

        if self.client_parameters is not None:
            with open(output_directory.joinpath("client_parameters.json"), "wb") as f:
                f.write(self.client_parameters)

        # pylint: enable=too-many-branches


class DebugArtifacts:
    """
    DebugArtifacts class, to export information about the compilation process for single function.
    """

    module_artifacts: ModuleDebugArtifacts

    def __init__(self, output_directory: Union[str, Path] = DEFAULT_OUTPUT_DIRECTORY):
        self.module_artifacts = ModuleDebugArtifacts([], output_directory)

    def export(self):
        """
        Export the collected information to `self.output_directory`.
        """
        self.module_artifacts.export()

    @property
    def output_directory(self) -> Path:  # pragma: no cover
        """
        Return the directory to export artifacts to.
        """
        return self.module_artifacts.output_directory

    @property
    def mlir_to_compile(self) -> Optional[str]:
        """
        Return the mlir string.
        """
        return self.module_artifacts.mlir_to_compile
