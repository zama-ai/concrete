"""
Declaration of `Circuit` class.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
from concrete.compiler import (
    ClientParameters,
    ClientSupport,
    CompilationOptions,
    JITCompilationResult,
    JITLambda,
    JITSupport,
    KeySet,
    KeySetCache,
    PublicArguments,
    PublicResult,
)

from ..dtypes import Integer
from ..internal.utils import assert_that
from ..representation import Graph
from ..values import Value
from .configuration import CompilationConfiguration


class Circuit:
    """
    Circuit class, to combine computation graph and compiler engine into a single object.
    """

    graph: Graph
    mlir: str
    virtual: bool

    _jit_support: JITSupport
    _compilation_result: JITCompilationResult

    _client_parameters: ClientParameters

    _keyset_cache: KeySetCache
    _keyset: KeySet

    _server_lambda: JITLambda

    def __init__(
        self,
        graph: Graph,
        mlir: str,
        configuration: Optional[CompilationConfiguration] = None,
        virtual: bool = False,
    ):
        configuration = configuration if configuration is not None else CompilationConfiguration()

        self.graph = graph
        self.mlir = mlir

        self.virtual = virtual
        if self.virtual:
            return

        options = CompilationOptions.new("main")

        options.set_loop_parallelize(configuration.loop_parallelize)
        options.set_dataflow_parallelize(configuration.dataflow_parallelize)
        options.set_auto_parallelize(configuration.auto_parallelize)

        self._jit_support = JITSupport.new()
        self._compilation_result = self._jit_support.compile(mlir, options)

        self._client_parameters = self._jit_support.load_client_parameters(self._compilation_result)

        self._keyset_cache = None
        if configuration.use_insecure_key_cache:
            assert_that(configuration.enable_unsafe_features)
            location = CompilationConfiguration.insecure_key_cache_location()
            if location is not None:
                self._keyset_cache = KeySetCache.new(str(location))
        self._keyset = None

        self._server_lambda = self._jit_support.load_server_lambda(self._compilation_result)

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

        if self.virtual:
            raise RuntimeError("Virtual circuits cannot use `keygen` method")

        if self._keyset is None or force:
            self._keyset = ClientSupport.key_set(self._client_parameters, self._keyset_cache)

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

        if self.virtual:
            raise RuntimeError("Virtual circuits cannot use `encrypt` method")

        if len(args) != len(self.graph.input_nodes):
            raise ValueError(f"Expected {len(self.graph.input_nodes)} inputs but got {len(args)}")

        sanitized_args = {}
        for index, node in self.graph.input_nodes.items():
            arg = args[index]
            is_valid = isinstance(arg, (int, np.integer)) or (
                isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.integer)
            )

            expected_value = node.output

            assert_that(isinstance(expected_value.dtype, Integer))
            expected_dtype = cast(Integer, expected_value.dtype)

            if is_valid:
                expected_min = expected_dtype.min()
                expected_max = expected_dtype.max()
                expected_shape = expected_value.shape

                actual_min = arg if isinstance(arg, int) else arg.min()
                actual_max = arg if isinstance(arg, int) else arg.max()
                actual_shape = () if isinstance(arg, int) else arg.shape

                is_valid = (
                    actual_min >= expected_min
                    and actual_max <= expected_max
                    and actual_shape == expected_shape
                )

                if is_valid:
                    sanitized_args[index] = arg if isinstance(arg, int) else arg.astype(np.uint8)

            if not is_valid:
                actual_value = Value.of(arg, is_encrypted=expected_value.is_encrypted)
                raise ValueError(
                    f"Expected argument {index} to be {expected_value} but it's {actual_value}"
                )

        self.keygen(force=False)
        return ClientSupport.encrypt_arguments(
            self._client_parameters,
            self._keyset,
            [sanitized_args[i] for i in range(len(sanitized_args))],
        )

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

        if self.virtual:
            raise RuntimeError("Virtual circuits cannot use `run` method")

        return self._jit_support.server_call(self._server_lambda, args)

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

        if self.virtual:
            raise RuntimeError("Virtual circuits cannot use `decrypt` method")

        results = ClientSupport.decrypt_result(self._keyset, result)
        if not isinstance(results, tuple):
            results = (results,)

        sanitized_results: List[Union[int, np.ndarray]] = []

        for index, node in self.graph.output_nodes.items():
            expected_value = node.output
            assert_that(isinstance(expected_value.dtype, Integer))

            expected_dtype = cast(Integer, expected_value.dtype)
            n = expected_dtype.bit_width

            result = results[index] % (2 ** n)
            if expected_dtype.is_signed:
                if isinstance(result, int):
                    sanititzed_result = result if result < (2 ** (n - 1)) else result - (2 ** n)
                    sanitized_results.append(sanititzed_result)
                else:
                    result = result.astype(np.longlong)  # to prevent overflows in numpy
                    sanititzed_result = np.where(result < (2 ** (n - 1)), result, result - (2 ** n))
                    sanitized_results.append(sanititzed_result.astype(np.int8))
            else:
                sanitized_results.append(
                    result if isinstance(result, int) else result.astype(np.uint8)
                )

        return sanitized_results[0] if len(sanitized_results) == 1 else tuple(sanitized_results)

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

        if self.virtual:
            return self.graph(*args)

        return self.decrypt(self.run(self.encrypt(*args)))
