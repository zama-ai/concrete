#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""ServerCircuit."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    ServerCircuit as _ServerCircuit,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp
from .public_arguments import PublicArguments
from .public_result import PublicResult
from .evaluation_keys import EvaluationKeys


class ServerCircuit(WrapperCpp):
    """ServerCircuit references a circuit that can be called for execution and simulation."""

    def __init__(self, server_circuit: _ServerCircuit):
        """Wrap the native Cpp object.

        Args:
            server_circuit (_ServerCircuit): object to wrap

        Raises:
            TypeError: if server_circuit is not of type _ServerCircuit
        """
        if not isinstance(server_circuit, _ServerCircuit):
            raise TypeError(
                f"server_circuit must be of type _ServerCircuit, not {type(server_circuit)}"
            )
        super().__init__(server_circuit)

    def call(
        self,
        public_arguments: PublicArguments,
        evaluation_keys: EvaluationKeys,
    ) -> PublicResult:
        """Executes the circuit on the public arguments.

        Args:
            public_arguments (PublicArguments): public arguments to execute on
            execution_keys (EvaluationKeys): evaluation keys to use for execution.

        Raises:
            TypeError: if public_arguments is not of type PublicArguments, or if evaluation_keys is
                not of type EvaluationKeys

        Returns:
            PublicResult: A public result object containing the results.
        """
        if not isinstance(public_arguments, PublicArguments):
            raise TypeError(
                f"public_arguments must be of type PublicArguments, not "
                f"{type(public_arguments)}"
            )
        if not isinstance(evaluation_keys, EvaluationKeys):
            raise TypeError(
                f"simulation must be of type EvaluationKeys, not "
                f"{type(evaluation_keys)}"
            )
        return PublicResult.wrap(
            self.cpp().call(public_arguments.cpp(), evaluation_keys.cpp())
        )

    def simulate(
        self,
        public_arguments: PublicArguments,
    ) -> PublicResult:
        """Simulates the circuit on the public arguments.

        Args:
            public_arguments (PublicArguments): public arguments to execute on

        Raises:
            TypeError: if public_arguments is not of type PublicArguments

        Returns:
            PublicResult: A public result object containing the results.
        """
        if not isinstance(public_arguments, PublicArguments):
            raise TypeError(
                f"public_arguments must be of type PublicArguments, not "
                f"{type(public_arguments)}"
            )
        return PublicResult.wrap(self.cpp().simulate(public_arguments.cpp()))
