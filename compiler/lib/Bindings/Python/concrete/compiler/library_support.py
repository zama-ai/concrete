#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

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
from .compilation_feedback import CompilationFeedback
from .wrapper import WrapperCpp
from .utils import lookup_runtime_lib
from .evaluation_keys import EvaluationKeys


# Default output path for compilation artifacts
DEFAULT_OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.curdir, "concrete-compiler_compilation_artifacts")
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
        self.output_dir_path = DEFAULT_OUTPUT_PATH

    @property
    def output_dir_path(self) -> str:
        """Path where to store compilation artifacts."""
        return self._output_dir_path

    @output_dir_path.setter
    def output_dir_path(self, path: str):
        if not isinstance(path, str):
            raise TypeError(f"path must be of type str, not {type(path)}")
        self._output_dir_path = path

    @staticmethod
    # pylint: disable=arguments-differ
    def new(
        output_path: str = DEFAULT_OUTPUT_PATH,
        runtime_library_path: Optional[str] = None,
        generateSharedLib: bool = True,
        generateStaticLib: bool = False,
        generateClientParameters: bool = True,
        generateCompilationFeedback: bool = True,
        generateCppHeader: bool = False,
    ) -> "LibrarySupport":
        """Build a LibrarySupport.

        Args:
            output_path (str, optional): path where to store compilation artifacts.
                Defaults to DEFAULT_OUTPUT_PATH.
            runtime_library_path (Optional[str], optional): path to the runtime library. Defaults to None.
            generateSharedLib (bool): whether to emit shared library or not. Default to True.
            generateStaticLib (bool): whether to emit static library or not. Default to False.
            generateClientParameters (bool): whether to emit client parameters or not. Default to True.
            generateCppHeader (bool): whether to emit cpp header or not. Default to False.

        Raises:
            TypeError: if output_path is not of type str
            TypeError: if runtime_library_path is not of type str
            TypeError: if one of the generation flags is not of type bool

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
        for name, value in [
            ("generateSharedLib", generateSharedLib),
            ("generateStaticLib", generateStaticLib),
            ("generateClientParameters", generateClientParameters),
            ("generateCompilationFeedback", generateCompilationFeedback),
            ("generateCppHeader", generateCppHeader),
        ]:
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be of type bool, not {type(value)}")
        library_support = LibrarySupport.wrap(
            _LibrarySupport(
                output_path,
                runtime_library_path,
                generateSharedLib,
                generateStaticLib,
                generateClientParameters,
                generateCompilationFeedback,
                generateCppHeader,
            )
        )
        library_support.output_dir_path = output_path
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
        """Reload the library compilation result from the output_dir_path.

        Args:
            func_name: entrypoint function name

        Returns:
            LibraryCompilationResult: loaded library
        """
        if not isinstance(func_name, str):
            raise TypeError(f"func_name must be of type str, not {type(func_name)}")
        return LibraryCompilationResult.new(self.output_dir_path, func_name)

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

    def load_compilation_feedback(
        self, compilation_result: LibraryCompilationResult
    ) -> CompilationFeedback:
        """Load the compilation feedback from the JIT compilation result.

        Args:
            compilation_result (JITCompilationResult): result of the JIT compilation

        Raises:
            TypeError: if compilation_result is not of type JITCompilationResult

        Returns:
            CompilationFeedback: the compilation feedback for the compiled program
        """
        if not isinstance(compilation_result, LibraryCompilationResult):
            raise TypeError(
                f"compilation_result must be of type JITCompilationResult, not {type(compilation_result)}"
            )
        return CompilationFeedback.wrap(
            self.cpp().load_compilation_feedback(compilation_result.cpp())
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
        self,
        library_lambda: LibraryLambda,
        public_arguments: PublicArguments,
        evaluation_keys: EvaluationKeys,
    ) -> PublicResult:
        """Call the library with public_arguments.

        Args:
            library_lambda (LibraryLambda): reference to the compiled library
            public_arguments (PublicArguments): arguments to use for execution
            evaluation_keys (EvaluationKeys): evaluation keys to use for execution

        Raises:
            TypeError: if library_lambda is not of type LibraryLambda
            TypeError: if public_arguments is not of type PublicArguments
            TypeError: if evaluation_keys is not of type EvaluationKeys

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
        if not isinstance(evaluation_keys, EvaluationKeys):
            raise TypeError(
                f"evaluation_keys must be of type EvaluationKeys, not {type(evaluation_keys)}"
            )
        return PublicResult.wrap(
            self.cpp().server_call(
                library_lambda.cpp(),
                public_arguments.cpp(),
                evaluation_keys.cpp(),
            )
        )

    def get_shared_lib_path(self) -> str:
        """Get the path where the shared library is expected to be.

        Returns:
            str: path to the shared library
        """
        return self.cpp().get_shared_lib_path()

    def get_client_parameters_path(self) -> str:
        """Get the path where the client parameters file is expected to be.

        Returns:
            str: path to the client parameters file
        """
        return self.cpp().get_client_parameters_path()
