#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""ServerProgram."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    ServerProgram as _ServerProgram,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp
from .library_support import LibrarySupport
from .server_circuit import ServerCircuit


class ServerProgram(WrapperCpp):
    """ServerProgram references compiled circuit objects."""

    def __init__(self, server_program: _ServerProgram):
        """Wrap the native Cpp object.

        Args:
            server_program (_ServerProgram): object to wrap

        Raises:
            TypeError: if server_program is not of type _ServerProgram
        """
        if not isinstance(server_program, _ServerProgram):
            raise TypeError(
                f"server_program must be of type _ServerProgram, not {type(server_program)}"
            )
        super().__init__(server_program)

    @staticmethod
    def load(
        library_support: LibrarySupport,
        simulation: bool,
    ) -> "ServerProgram":
        """Loads the server program from a library support.

        Args:
            library_support (LibrarySupport): library support
            simulation (bool): use simulation for execution

        Raises:
            TypeError: if library_support is not of type LibrarySupport, or if simulation is not of type bool

        Returns:
            ServerProgram: A server program object containing references to circuits for calls.
        """
        if not isinstance(library_support, LibrarySupport):
            raise TypeError(
                f"library_support must be of type LibrarySupport, not "
                f"{type(library_support)}"
            )
        if not isinstance(simulation, bool):
            raise TypeError(
                f"simulation must be of type bool, not " f"{type(simulation)}"
            )
        return ServerProgram.wrap(
            _ServerProgram.load(library_support.cpp(), simulation)
        )

    def get_server_circuit(self, circuit_name: str) -> ServerCircuit:
        """Returns a given circuit if it is part of the program.

        Args:
            circuit_name (str): name of the circuit to retrieve.

        Raises:
            TypeError: if circuit_name is not of type str
            RuntimeError: if the circuit is not part of the program
        """
        if not isinstance(circuit_name, str):
            raise TypeError(
                f"circuit_name must be of type str, not {type(circuit_name)}"
            )

        return ServerCircuit.wrap(self.cpp().get_server_circuit(circuit_name))
