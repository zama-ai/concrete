"""
Declaration of `Circuit` class.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
from concrete.compiler import PublicArguments, PublicResult

from ..dtypes import Integer
from ..internal.utils import assert_that
from ..representation import Graph
from .client import Client
from .configuration import Configuration
from .server import Server


class Circuit:
    """
    Circuit class, to combine computation graph, mlir, client and server into a single object.
    """

    configuration: Configuration

    graph: Graph
    mlir: str

    client: Client
    server: Server

    def __init__(self, graph: Graph, mlir: str, configuration: Optional[Configuration] = None):
        self.configuration = configuration if configuration is not None else Configuration()

        self.graph = graph
        self.mlir = mlir

        if self.configuration.virtual:
            assert_that(self.configuration.enable_unsafe_features)
            return

        input_signs = []
        for i in range(len(graph.input_nodes)):  # pylint: disable=consider-using-enumerate
            input_value = graph.input_nodes[i].output
            assert_that(isinstance(input_value.dtype, Integer))
            input_dtype = cast(Integer, input_value.dtype)
            input_signs.append(input_dtype.is_signed)

        output_signs = []
        for i in range(len(graph.output_nodes)):  # pylint: disable=consider-using-enumerate
            output_value = graph.output_nodes[i].output
            assert_that(isinstance(output_value.dtype, Integer))
            output_dtype = cast(Integer, output_value.dtype)
            output_signs.append(output_dtype.is_signed)

        self.server = Server.create(mlir, input_signs, output_signs, self.configuration)

        keyset_cache_directory = None
        if self.configuration.use_insecure_key_cache:
            assert_that(self.configuration.enable_unsafe_features)
            assert_that(self.configuration.insecure_key_cache_location is not None)
            keyset_cache_directory = self.configuration.insecure_key_cache_location

        self.client = Client(self.server.client_specs, keyset_cache_directory)

    def __str__(self):
        return self.graph.format()

    def draw(
        self,
        show: bool = False,
        horizontal: bool = False,
        save_to: Optional[Union[Path, str]] = None,
    ) -> Path:
        """
        Draw `self.graph` and optionally save/show the drawing.

        note that this function requires the python `pygraphviz` package
        which itself requires the installation of `graphviz` packages
        see https://pygraphviz.github.io/documentation/stable/install.html

        Args:
            show (bool, default = False):
                whether to show the drawing using matplotlib or not

            horizontal (bool, default = False):
                whether to draw horizontally or not

            save_to (Optional[Path], default = None):
                path to save the drawing
                a temporary file will be used if it's None

        Returns:
            Path:
                path to the saved drawing
        """

        return self.graph.draw(show, horizontal, save_to)

    def keygen(self, force: bool = False):
        """
        Generate keys required for homomorphic evaluation.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated
        """

        if self.configuration.virtual:
            raise RuntimeError("Virtual circuits cannot use `keygen` method")

        self.client.keygen(force)

    def encrypt(self, *args: Union[int, np.ndarray]) -> PublicArguments:
        """
        Prepare inputs to be run on the circuit.

        Args:
            *args (Union[int, numpy.ndarray]):
                inputs to the circuit

        Returns:
            PublicArguments:
                encrypted and plain arguments as well as public keys
        """

        if self.configuration.virtual:
            raise RuntimeError("Virtual circuits cannot use `encrypt` method")

        return self.client.encrypt(*args)

    def run(self, args: PublicArguments) -> PublicResult:
        """
        Evaluate circuit using encrypted arguments.

        Args:
            args (PublicArguments):
                arguments to the circuit (can be obtained with `encrypt` method of `Circuit`)

        Returns:
            PublicResult:
                encrypted result of homomorphic evaluaton
        """

        if self.configuration.virtual:
            raise RuntimeError("Virtual circuits cannot use `run` method")

        self.keygen(force=False)
        return self.server.run(args, self.client.evaluation_keys)

    def decrypt(
        self,
        result: PublicResult,
    ) -> Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
        """
        Decrypt result of homomorphic evaluaton.

        Args:
            result (PublicResult):
                encrypted result of homomorphic evaluaton

        Returns:
            Union[int, numpy.ndarray]:
                clear result of homomorphic evaluaton
        """

        if self.configuration.virtual:
            raise RuntimeError("Virtual circuits cannot use `decrypt` method")

        return self.client.decrypt(result)

    def encrypt_run_decrypt(self, *args: Any) -> Any:
        """
        Encrypt inputs, run the circuit, and decrypt the outputs in one go.

        Args:
            *args (Union[int, numpy.ndarray]):
                inputs to the circuit

        Returns:
            Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
                clear result of homomorphic evaluation
        """

        if self.configuration.virtual:
            return self.graph(*args)

        return self.decrypt(self.run(self.encrypt(*args)))

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """

        self.server.cleanup()
