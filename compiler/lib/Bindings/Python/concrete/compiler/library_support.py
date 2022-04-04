#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

"""LibrarySupport.

Library support provides a way to compile an MLIR program into a library that can be later loaded
to execute the compiled code.
"""
import os
from typing import Optional

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    LibrarySupport as _LibrarySupport,
)

# pylint: enable=no-name-in-module,import-error
from .compilation_options import CompilationOptions
from .library_compilation_result import LibraryCompilationResult
from .public_arguments import PublicArguments
from .library_lambda import LibraryLambda
from .public_result import PublicResult
from .client_parameters import ClientParameters
from .wrapper import WrapperCpp
from .utils import lookup_runtime_lib


# Default output path for compiled libraries
DEFAULT_OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.curdir, "concrete-compiler_output_lib")
)


class LibrarySupport(WrapperCpp):
    """Support class for library compilation and execution."""

    def __init__(self, library_support: _LibrarySupport):
        """Wrap the native Cpp object.

        Args:
            library_support (_LibrarySupport): object to wrap

        Raises:
            TypeError: if library_support is not of type _LibrarySupport
        """
        if not isinstance(library_support, _LibrarySupport):
            raise TypeError(
                f"library_support must be of type _LibrarySupport, not "
                f"{type(library_support)}"
            )
        super().__init__(library_support)
        self.library_path = DEFAULT_OUTPUT_PATH

    @property
    def library_path(self) -> str:
        """Path where to store compiled libraries."""
        return self._library_path

    @library_path.setter
    def library_path(self, path: str):
        if not isinstance(path, str):
            raise TypeError(f"path must be of type str, not {type(path)}")
        self._library_path = path

    @staticmethod
    # pylint: disable=arguments-differ
    def new(
        output_path: str = DEFAULT_OUTPUT_PATH,
        runtime_library_path: Optional[str] = None,
    ) -> "LibrarySupport":
        """Build a LibrarySupport.

        Args:
            output_path (str, optional): path where to store compiled libraries.
                Defaults to DEFAULT_OUTPUT_PATH.
            runtime_library_path (Optional[str], optional): path to the runtime library. Defaults to None.

        Raises:
            TypeError: if output_path is not of type str
            TypeError: if runtime_library_path is not of type str

        Returns:
            LibrarySupport
        """
        if runtime_library_path is None:
            runtime_library_path = lookup_runtime_lib()
        if not isinstance(output_path, str):
            raise TypeError(f"output_path must be of type str, not {type(output_path)}")
        if not isinstance(runtime_library_path, str):
            raise TypeError(
                f"runtime_library_path must be of type str, not {type(runtime_library_path)}"
            )
        library_support = LibrarySupport.wrap(
            _LibrarySupport(output_path, runtime_library_path)
        )
        library_support.library_path = output_path
        return library_support

    def compile(
        self,
        mlir_program: str,
        options: CompilationOptions = CompilationOptions.new("main"),
    ) -> LibraryCompilationResult:
        """Compile an MLIR program using Concrete dialects into a library.

        Args:
            mlir_program (str): textual representation of the mlir program to compile
            options (CompilationOptions): compilation options

        Raises:
            TypeError: if mlir_program is not of type str
            TypeError: if options is not of type CompilationOptions

        Returns:
            LibraryCompilationResult: the result of the library compilation
        """
        if not isinstance(mlir_program, str):
            raise TypeError(
                f"mlir_program must be of type str, not {type(mlir_program)}"
            )
        if not isinstance(options, CompilationOptions):
            raise TypeError(
                f"options must be of type CompilationOptions, not {type(options)}"
            )
        return LibraryCompilationResult.wrap(
            self.cpp().compile(mlir_program, options.cpp())
        )

    def reload(self, func_name: str = "main") -> LibraryCompilationResult:
        """Reload the library compilation result from the library_path.

        Args:
            func_name: entrypoint function name

        Returns:
            LibraryCompilationResult: loaded library
        """
        if not isinstance(func_name, str):
            raise TypeError(f"func_name must be of type str, not {type(func_name)}")
        return LibraryCompilationResult.new(self.library_path, func_name)

    def load_client_parameters(
        self, library_compilation_result: LibraryCompilationResult
    ) -> ClientParameters:
        """Load the client parameters from the library compilation result.

        Args:
            library_compilation_result (LibraryCompilationResult): compilation result of the library

        Raises:
            TypeError: if library_compilation_result is not of type LibraryCompilationResult

        Returns:
            ClientParameters: appropriate client parameters for the compiled library
        """
        if not isinstance(library_compilation_result, LibraryCompilationResult):
            raise TypeError(
                f"library_compilation_result must be of type LibraryCompilationResult, not "
                f"{type(library_compilation_result)}"
            )

        return ClientParameters.wrap(
            self.cpp().load_client_parameters(library_compilation_result.cpp())
        )

    def load_server_lambda(
        self, library_compilation_result: LibraryCompilationResult
    ) -> LibraryLambda:
        """Load the server lambda from the library compilation result.

        Args:
            library_compilation_result (LibraryCompilationResult): compilation result of the library

        Raises:
            TypeError: if library_compilation_result is not of type LibraryCompilationResult

        Returns:
            LibraryLambda: executable reference to the library
        """
        if not isinstance(library_compilation_result, LibraryCompilationResult):
            raise TypeError(
                f"library_compilation_result must be of type LibraryCompilationResult, not "
                f"{type(library_compilation_result)}"
            )
        return LibraryLambda.wrap(
            self.cpp().load_server_lambda(library_compilation_result.cpp())
        )

    def server_call(
        self, library_lambda: LibraryLambda, public_arguments: PublicArguments
    ) -> PublicResult:
        """Call the library with public_arguments.

        Args:
            library_lambda (LibraryLambda): reference to the compiled library
            public_arguments (PublicArguments): arguments to use for execution

        Raises:
            TypeError: if library_lambda is not of type LibraryLambda
            TypeError: if public_arguments is not of type PublicArguments

        Returns:
            PublicResult: result of the execution
        """
        if not isinstance(library_lambda, LibraryLambda):
            raise TypeError(
                f"library_lambda must be of type LibraryLambda, not {type(library_lambda)}"
            )
        if not isinstance(public_arguments, PublicArguments):
            raise TypeError(
                f"public_arguments must be of type PublicArguments, not {type(public_arguments)}"
            )
        return PublicResult.wrap(
            self.cpp().server_call(library_lambda.cpp(), public_arguments.cpp())
        )
