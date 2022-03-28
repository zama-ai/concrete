"""Module to hold the result of compilation."""

from pathlib import Path
from typing import Optional, Union

import numpy
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

from .debugging import draw_graph, format_operation_graph
from .operator_graph import OPGraph


class FHECircuit:
    """Class which is the result of compilation."""

    op_graph: OPGraph
    _jit_support: JITSupport
    _compilation_result: JITCompilationResult
    _client_parameters: ClientParameters
    _server_lambda: JITLambda
    _keyset_cache: KeySetCache
    _keyset: KeySet

    def __init__(
        self,
        op_graph: OPGraph,
        mlir_str: str,
        unsecure_key_set_cache_path: Optional[str] = None,
        auto_parallelize: bool = False,
        loop_parallelize: bool = False,
        dataflow_parallelize: bool = False,
    ):
        self.op_graph = op_graph
        self._jit_support = JITSupport.new()
        # Set compilation options
        options = CompilationOptions.new("main")
        options.set_auto_parallelize(auto_parallelize)
        options.set_loop_parallelize(loop_parallelize)
        options.set_dataflow_parallelize(dataflow_parallelize)
        # Compile
        self._compilation_result = self._jit_support.compile(mlir_str, options)
        self._client_parameters = self._jit_support.load_client_parameters(self._compilation_result)
        self._server_lambda = self._jit_support.load_server_lambda(self._compilation_result)
        # Setup keyset cache
        self._keyset_cache = None
        if unsecure_key_set_cache_path:
            self._keyset_cache = KeySetCache.new(unsecure_key_set_cache_path)
        self._keyset = None

    def __str__(self):
        return format_operation_graph(self.op_graph)

    def draw(
        self,
        show: bool = False,
        vertical: bool = True,
        save_to: Optional[Path] = None,
    ) -> str:
        """Draw operation graph of the circuit and optionally save/show the drawing.

        Args:
            show (bool): if set to True, the drawing will be shown using matplotlib
            vertical (bool): if set to True, the orientation will be vertical
            save_to (Optional[Path]): if specified, the drawn graph will be saved to this path;
                otherwise it will be saved to a temporary file

        Returns:
            str: path of the file where the drawn graph is saved

        """

        return draw_graph(self.op_graph, show, vertical, save_to)

    def keygen(self, force: bool = False):
        """Generate the keys required for the encrypted circuit.

        Args:
            force (bool, optional): generate even if keyset already exists. Defaults to False.
        """
        if self._keyset is None or force:
            self._keyset = ClientSupport.key_set(self._client_parameters, self._keyset_cache)

    def encrypt(self, *args: Union[int, numpy.ndarray]) -> PublicArguments:
        """Encrypt the inputs of the circuit.

        Args:
            *args (Union[int, numpy.ndarray]): plain input of the circuit

        Returns:
            PublicArguments: encrypted and plain arguments as well as public keys
        """
        # Make sure keys are available: shouldn't regenerate if they already exist
        self.keygen(force=False)
        return ClientSupport.encrypt_arguments(self._client_parameters, self._keyset, args)

    def run(self, args: PublicArguments) -> PublicResult:
        """Evaluate the the encrypted circuit (no encryption or decryption involved).

        Args:
            args (PublicArguments): encrypted inputs to the circuit

        Returns:
            PublicResult: encrypted result
        """
        return self._jit_support.server_call(self._server_lambda, args)

    def decrypt(self, result: PublicResult) -> Union[int, numpy.ndarray]:
        """Decrypt the result of the circuit.

        Args:
            result (PublicResult): encrypted result of the circuit

        Returns:
            Union[int, numpy.ndarray]: plain result of the circuit
        """
        return ClientSupport.decrypt_result(self._keyset, result)

    def encrypt_run_decrypt(self, *args: Union[int, numpy.ndarray]) -> Union[int, numpy.ndarray]:
        """Encrypt, evaluate, and decrypt the inputs on the circuit.

        Generate keyset automatically if not yet done.

        Args:
            *args (Union[int, numpy.ndarray]): plain inputs of the circuit

        Returns:
            Union[int, numpy.ndarray]: plain result of the circuit
        """
        self.keygen(force=False)
        public_args = self.encrypt(*args)
        encrypted_result = self.run(public_args)
        decrypted_result = self.decrypt(encrypted_result)
        return decrypted_result
