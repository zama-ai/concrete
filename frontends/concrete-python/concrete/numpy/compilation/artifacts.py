"""
Declaration of `DebugArtifacts` class.
"""

import inspect
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from ..representation import Graph

DEFAULT_OUTPUT_DIRECTORY: Path = Path(".artifacts")


class DebugArtifacts:
    """
    DebugArtifacts class, to export information about the compilation process.
    """

    output_directory: Path

    source_code: Optional[str]
    parameter_encryption_statuses: Dict[str, str]
    textual_representations_of_graphs: Dict[str, List[str]]
    final_graph: Optional[Graph]
    mlir_to_compile: Optional[str]
    client_parameters: Optional[bytes]

    def __init__(self, output_directory: Union[str, Path] = DEFAULT_OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)

        self.source_code = None
        self.parameter_encryption_statuses = {}
        self.textual_representations_of_graphs = {}
        self.final_graph = None
        self.mlir_to_compile = None
        self.client_parameters = None

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

    def add_mlir_to_compile(self, mlir: str):
        """
        Add textual representation of the resulting MLIR.

        Args:
            mlir (str):
                textual representation of the resulting MLIR
        """

        self.mlir_to_compile = mlir

    def add_client_parameters(self, client_parameters: bytes):
        """
        Add client parameters used.

        Args:
            client_parameters (bytes): client parameters
        """

        self.client_parameters = client_parameters

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
            # example `pip list` output

            # Package                       Version
            # ----------------------------- ---------
            # alabaster                     0.7.12
            # appdirs                       1.4.4
            # ...                           ...
            # ...                           ...
            # wrapt                         1.12.1
            # zipp                          3.5.0

            pip_process = subprocess.run(
                ["pip", "--disable-pip-version-check", "list"], stdout=subprocess.PIPE, check=True
            )
            dependencies = iter(pip_process.stdout.decode("utf-8").split("\n"))

            # skip 'Package ... Version' line
            next(dependencies)

            # skip '------- ... -------' line
            next(dependencies)

            for dependency in dependencies:
                tokens = [token for token in dependency.split(" ") if token != ""]  # noqa: S105
                if len(tokens) == 0:
                    continue

                name = tokens[0]
                version = tokens[1]

                f.write(f"{name}=={version}\n")

        if self.source_code is not None:
            with open(output_directory.joinpath("function.txt"), "w", encoding="utf-8") as f:
                f.write(self.source_code)

        if len(self.parameter_encryption_statuses) > 0:
            with open(output_directory.joinpath("parameters.txt"), "w", encoding="utf-8") as f:
                for name, parameter in self.parameter_encryption_statuses.items():
                    f.write(f"{name} :: {parameter}\n")

        identifier = 0

        textual_representations = self.textual_representations_of_graphs.items()
        for name, representations in textual_representations:
            for representation in representations:
                identifier += 1
                output_path = output_directory.joinpath(f"{identifier}.{name}.graph.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"{representation}\n")

        if self.mlir_to_compile is not None:
            assert self.final_graph is not None
            with open(output_directory.joinpath("mlir.txt"), "w", encoding="utf-8") as f:
                f.write(f"{self.mlir_to_compile}\n")

        if self.client_parameters is not None:
            with open(output_directory.joinpath("client_parameters.json"), "wb") as f:
                f.write(self.client_parameters)

        # pylint: enable=too-many-branches
