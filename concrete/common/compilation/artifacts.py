"""Module for compilation artifacts."""

import inspect
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import networkx as nx
from loguru import logger

from ..debugging import assert_true, draw_graph, format_operation_graph
from ..operator_graph import OPGraph
from ..representation.intermediate import IntermediateNode
from ..values import BaseValue

DEFAULT_OUTPUT_DIRECTORY: Path = Path(".artifacts")


class CompilationArtifacts:
    """Class that conveys information about compilation process."""

    output_directory: Path

    source_code_of_the_function_to_compile: Optional[str]
    parameters_of_the_function_to_compile: Dict[str, str]

    drawings_of_operation_graphs: Dict[str, str]
    textual_representations_of_operation_graphs: Dict[str, str]

    final_operation_graph: Optional[OPGraph]
    bounds_of_the_final_operation_graph: Optional[Dict[IntermediateNode, Dict[str, Any]]]
    mlir_of_the_final_operation_graph: Optional[str]

    def __init__(self, output_directory: Union[Path, str] = DEFAULT_OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)

        self.source_code_of_the_function_to_compile = None
        self.parameters_of_the_function_to_compile = {}

        self.drawings_of_operation_graphs = {}
        self.textual_representations_of_operation_graphs = {}

        self.final_operation_graph = None
        self.bounds_of_the_final_operation_graph = None
        self.mlir_of_the_final_operation_graph = None

    def add_function_to_compile(self, function: Union[Callable, str]):
        """Add the function to compile to artifacts.

        Args:
            function (Union[Callable, str]): the function to compile or source code of it

        Returns:
            None
        """

        try:
            self.source_code_of_the_function_to_compile = (
                function if isinstance(function, str) else inspect.getsource(function)
            )
        # When using the python console we cannot use getsource, so catch that and emit an error
        except OSError:  # pragma: no cover
            function_str = function if isinstance(function, str) else function.__name__
            logger.error(f"Could not get source for function: {function_str}")
            self.source_code_of_the_function_to_compile = "unavailable"

    def add_parameter_of_function_to_compile(self, name: str, value: Union[BaseValue, str]):
        """Add a parameter of the function to compile to the artifacts.

        Args:
            name (str): name of the parameter
            value (Union[BaseValue, str]): value of the parameter or textual representation of it

        Returns:
            None
        """

        self.parameters_of_the_function_to_compile[name] = str(value)

    def add_operation_graph(self, name: str, operation_graph: OPGraph):
        """Add an operation graph to the artifacts.

        Args:
            name (str): name of the graph
            operation_graph (OPGraph): the operation graph itself

        Returns:
            None
        """

        try:
            drawing = draw_graph(operation_graph)
            self.drawings_of_operation_graphs[name] = drawing
        # Do not crash on imports ourselves for drawings if the package is not installed
        except ImportError as e:  # pragma: no cover
            if "pygraphviz" in str(e):
                pass
            else:
                raise e
        textual_representation = format_operation_graph(operation_graph)

        self.textual_representations_of_operation_graphs[name] = textual_representation

        self.final_operation_graph = operation_graph

    def add_final_operation_graph_bounds(self, bounds: Dict[IntermediateNode, Dict[str, Any]]):
        """Add the bounds of the final operation graph to the artifacts.

        Args:
            bounds (Dict[IntermediateNode, Dict[str, Any]]): the bound dictionary

        Returns:
            None
        """

        assert_true(self.final_operation_graph is not None)
        self.bounds_of_the_final_operation_graph = bounds

    def add_final_operation_graph_mlir(self, mlir: str):
        """Add the mlir of the final operation graph to the artifacts.

        Args:
            mlir (str): the mlir code of the final operation graph

        Returns:
            None
        """

        assert_true(self.final_operation_graph is not None)
        self.mlir_of_the_final_operation_graph = mlir

    def export(self):
        """Export the artifacts to a the output directory.

        Returns:
            None
        """

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
                tokens = [token for token in dependency.split(" ") if token != ""]
                if len(tokens) == 0:
                    continue

                name = tokens[0]
                version = tokens[1]

                f.write(f"{name}=={version}\n")

        if self.source_code_of_the_function_to_compile is not None:
            with open(output_directory.joinpath("function.txt"), "w", encoding="utf-8") as f:
                f.write(self.source_code_of_the_function_to_compile)

        if len(self.parameters_of_the_function_to_compile) > 0:
            with open(output_directory.joinpath("parameters.txt"), "w", encoding="utf-8") as f:
                for name, parameter in self.parameters_of_the_function_to_compile.items():
                    f.write(f"{name} :: {parameter}\n")

        drawings = self.drawings_of_operation_graphs.items()
        for index, (name, drawing_filename) in enumerate(drawings):
            identifier = CompilationArtifacts._identifier(index, name)
            shutil.copy(drawing_filename, output_directory.joinpath(f"{identifier}.png"))

        textual_representations = self.textual_representations_of_operation_graphs.items()
        for index, (name, representation) in enumerate(textual_representations):
            identifier = CompilationArtifacts._identifier(index, name)
            with open(output_directory.joinpath(f"{identifier}.txt"), "w", encoding="utf-8") as f:
                f.write(f"{representation}")

        if self.bounds_of_the_final_operation_graph is not None:
            assert_true(self.final_operation_graph is not None)
            with open(output_directory.joinpath("bounds.txt"), "w", encoding="utf-8") as f:
                # TODO:
                #   if nx.topological_sort is not deterministic between calls,
                #   the lines below will not work properly
                #   thus, we may want to change this in the future
                for index, node in enumerate(nx.topological_sort(self.final_operation_graph.graph)):
                    bounds = self.bounds_of_the_final_operation_graph.get(node)
                    assert_true(bounds is not None)
                    f.write(f"%{index} :: [{bounds.get('min')}, {bounds.get('max')}]\n")

        if self.mlir_of_the_final_operation_graph is not None:
            assert_true(self.final_operation_graph is not None)
            with open(output_directory.joinpath("mlir.txt"), "w", encoding="utf-8") as f:
                f.write(self.mlir_of_the_final_operation_graph)

    @staticmethod
    def _identifier(index, name):
        return f"{index + 1}.{name}.graph"
