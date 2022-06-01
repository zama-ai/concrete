"""
Declaration of `Server` class.
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from concrete.compiler import (
    CompilationOptions,
    EvaluationKeys,
    JITCompilationResult,
    JITLambda,
    JITSupport,
    LibraryCompilationResult,
    LibraryLambda,
    LibrarySupport,
    PublicArguments,
    PublicResult,
)

from ..internal.utils import assert_that
from .configuration import Configuration
from .specs import ClientSpecs


class Server:
    """
    Server class, which can be used to perform homomorphic computation.
    """

    client_specs: ClientSpecs

    _output_dir: Optional[tempfile.TemporaryDirectory]
    _support: Union[JITSupport, LibrarySupport]
    _compilation_result: Union[JITCompilationResult, LibraryCompilationResult]
    _server_lambda: Union[JITLambda, LibraryLambda]

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
        self._server_lambda = server_lambda

        assert_that(
            support.load_client_parameters(compilation_result).serialize()
            == client_specs.client_parameters.serialize()
        )

    @staticmethod
    def create(
        mlir: str,
        input_signs: List[bool],
        output_signs: List[bool],
        configuration: Configuration,
    ) -> "Server":
        """
        Create a server using MLIR and output sign information.

        Args:
            mlir (str):
                mlir to compile

            input_signs (List[bool]):
                sign status of the inputs

            output_signs (List[bool]):
                sign status of the outputs

            configuration (Optional[Configuration], default = None):
                configuration to use
        """

        options = CompilationOptions.new("main")

        options.set_loop_parallelize(configuration.loop_parallelize)
        options.set_dataflow_parallelize(configuration.dataflow_parallelize)
        options.set_auto_parallelize(configuration.auto_parallelize)
        options.set_p_error(configuration.p_error)

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
        client_specs = ClientSpecs(input_signs, client_parameters, output_signs)
        return Server(client_specs, output_dir, support, compilation_result, server_lambda)

    def save(self, path: Union[str, Path]):
        """
        Save the server into the given path in zip format.

        Args:
            path (Union[str, Path]):
                path to save the server
        """

        if self._output_dir is None:
            raise RuntimeError("Just-in-Time compilation cannot be saved")

        with open(Path(self._output_dir.name) / "client.specs.json", "w", encoding="utf-8") as f:
            f.write(self.client_specs.serialize())

        path = str(path)
        if path.endswith(".zip"):
            path = path[: len(path) - 4]

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

        with open(output_dir_path / "client.specs.json", "r", encoding="utf-8") as f:
            client_specs = ClientSpecs.unserialize(f.read())

        support = LibrarySupport.new(
            str(output_dir_path),
            generateCppHeader=False,
            generateStaticLib=False,
        )
        compilation_result = support.reload("main")
        server_lambda = support.load_server_lambda(compilation_result)

        return Server(client_specs, output_dir, support, compilation_result, server_lambda)

    def run(self, args: PublicArguments, evaluation_keys: EvaluationKeys) -> PublicResult:
        """
        Evaluate using encrypted arguments.

        Args:
            args (PublicArguments):
                encrypted arguments of the computation

            evaluation_keys (EvaluationKeys):
                evaluation keys for encrypted computation

        Returns:
            PublicResult:
                encrypted result of the computation
        """

        return self._support.server_call(self._server_lambda, args, evaluation_keys)

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """

        if self._output_dir is not None:
            self._output_dir.cleanup()
