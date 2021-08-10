"""Module for compilation artifacts"""

import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx

from ..debugging.draw_graph import get_printable_graph
from ..operator_graph import OPGraph
from ..representation import intermediate as ir


class CompilationArtifacts:
    """Class that conveys information about compilation process"""

    operation_graph: Optional[OPGraph]
    bounds: Optional[Dict[ir.IntermediateNode, Dict[str, Any]]]

    def __init__(self):
        self.operation_graph = None
        self.bounds = None

    def export(self, output_directory: Path):
        """Exports the artifacts in a textual format

        Args:
            output_directory (Path): the directory to save the artifacts

        Returns:
            None
        """

        with open(output_directory.joinpath("environment.txt"), "w") as f:
            f.write(f"{platform.platform()} {platform.version()}\n")
            f.write(f"Python {platform.python_version()}\n")

        with open(output_directory.joinpath("requirements.txt"), "w") as f:
            # example `pip list` output

            # Package                       Version
            # ----------------------------- ---------
            # alabaster                     0.7.12
            # appdirs                       1.4.4
            # ...                           ...
            # ...                           ...
            # wrapt                         1.12.1
            # zipp                          3.5.0

            pip_process = subprocess.run(["pip", "list"], stdout=subprocess.PIPE, check=True)
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

        if self.operation_graph is not None:
            with open(output_directory.joinpath("graph.txt"), "w") as f:
                f.write(f"{get_printable_graph(self.operation_graph)[1:]}\n")

            if self.bounds is not None:
                with open(output_directory.joinpath("bounds.txt"), "w") as f:
                    # TODO:
                    #   if nx.topological_sort is not deterministic between calls,
                    #   the lines below will not work properly
                    #   thus, we may want to change this in the future
                    for index, node in enumerate(nx.topological_sort(self.operation_graph.graph)):
                        bounds = self.bounds.get(node)
                        assert bounds is not None
                        f.write(f"%{index} :: [{bounds.get('min')}, {bounds.get('max')}]\n")
