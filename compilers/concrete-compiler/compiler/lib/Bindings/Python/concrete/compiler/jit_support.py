#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""JITSupport.

Just-in-time compilation provide a way to compile and execute an MLIR program while keeping the executable
code in memory.
"""

from typing import Optional

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    JITSupport as _JITSupport,
)

# pylint: enable=no-name-in-module,import-error
from .utils import lookup_runtime_lib
from .compilation_options import CompilationOptions
from .jit_compilation_result import JITCompilationResult
from .client_parameters import ClientParameters
from .compilation_feedback import CompilationFeedback
from .jit_lambda import JITLambda
from .public_arguments import PublicArguments
from .public_result import PublicResult
from .wrapper import WrapperCpp
from .evaluation_keys import EvaluationKeys


class JITSupport(WrapperCpp):
    """Support class for JIT compilation and execution."""

    def __init__(self, jit_support: _JITSupport):
        """Wrap the native Cpp object.

        Args:
            jit_support (_JITSupport): object to wrap

        Raises:
            TypeError: if jit_support is not of type _JITSupport
        """
        if not isinstance(jit_support, _JITSupport):
            raise TypeError(
                f"jit_support must be of type _JITSupport, not {type(jit_support)}"
            )
        super().__init__(jit_support)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(runtime_lib_path: Optional[str] = None) -> "JITSupport":
        """Build a JITSupport.

        Args:
            runtime_lib_path (Optional[str]): path to the runtime library. Defaults to None.

        Raises:
            TypeError: if runtime_lib_path is not of type str or None

        Returns:
            JITSupport
        """
        if runtime_lib_path is None:
            runtime_lib_path = lookup_runtime_lib()
        else:
            if not isinstance(runtime_lib_path, str):
                raise TypeError(
                    f"runtime_lib_path must be of type str, not {type(runtime_lib_path)}"
                )
        return JITSupport.wrap(_JITSupport(runtime_lib_path))

    # pylint: enable=arguments-differ

    def compile(
        self,
        mlir_program: str,
        options: CompilationOptions = CompilationOptions.new("main"),
    ) -> JITCompilationResult:
        """JIT compile an MLIR program using Concrete dialects.

        Args:
            mlir_program (str): textual representation of the mlir program to compile
            options (CompilationOptions): compilation options

        Raises:
            TypeError: if mlir_program is not of type str
            TypeError: if options is not of type CompilationOptions

        Returns:
            JITCompilationResult: the result of the JIT compilation
        """
        if not isinstance(mlir_program, str):
            raise TypeError(
                f"mlir_program must be of type str, not {type(mlir_program)}"
            )
        if not isinstance(options, CompilationOptions):
            raise TypeError(
                f"options must be of type CompilationOptions, not {type(options)}"
            )
        return JITCompilationResult.wrap(
            self.cpp().compile(mlir_program, options.cpp())
        )

    def load_client_parameters(
        self, compilation_result: JITCompilationResult
    ) -> ClientParameters:
        """Load the client parameters from the JIT compilation result.

        Args:
            compilation_result (JITCompilationResult): result of the JIT compilation

        Raises:
            TypeError: if compilation_result is not of type JITCompilationResult

        Returns:
            ClientParameters: appropriate client parameters for the compiled program
        """
        if not isinstance(compilation_result, JITCompilationResult):
            raise TypeError(
                f"compilation_result must be of type JITCompilationResult, not {type(compilation_result)}"
            )
        return ClientParameters.wrap(
            self.cpp().load_client_parameters(compilation_result.cpp())
        )

    def load_compilation_feedback(
        self, compilation_result: JITCompilationResult
    ) -> CompilationFeedback:
        """Load the compilation feedback from the JIT compilation result.

        Args:
            compilation_result (JITCompilationResult): result of the JIT compilation

        Raises:
            TypeError: if compilation_result is not of type JITCompilationResult

        Returns:
            CompilationFeedback: the compilation feedback for the compiled program
        """
        if not isinstance(compilation_result, JITCompilationResult):
            raise TypeError(
                f"compilation_result must be of type JITCompilationResult, not {type(compilation_result)}"
            )
        return CompilationFeedback.wrap(
            self.cpp().load_compilation_feedback(compilation_result.cpp())
        )

    def load_server_lambda(self, compilation_result: JITCompilationResult) -> JITLambda:
        """Load the JITLambda from the JIT compilation result.

        Args:
            compilation_result (JITCompilationResult): result of the JIT compilation.

        Raises:
            TypeError: if compilation_result is not of type JITCompilationResult

        Returns:
            JITLambda: loaded JITLambda to be executed
        """
        if not isinstance(compilation_result, JITCompilationResult):
            raise TypeError(
                f"compilation_result must be a JITCompilationResult not {type(compilation_result)}"
            )
        return JITLambda.wrap(self.cpp().load_server_lambda(compilation_result.cpp()))

    def server_call(
        self,
        jit_lambda: JITLambda,
        public_arguments: PublicArguments,
        evaluation_keys: EvaluationKeys,
    ) -> PublicResult:
        """Call the JITLambda with public_arguments.

        Args:
            jit_lambda (JITLambda): A server lambda to call.
            public_arguments (PublicArguments): The arguments of the call.
            evaluation_keys (EvaluationKeys): Evalutation keys of the call.

        Raises:
            TypeError: if jit_lambda is not of type JITLambda
            TypeError: if public_arguments is not of type PublicArguments
            TypeError: if evaluation_keys is not of type EvaluationKeys

        Returns:
            PublicResult: the result of the call of the server lambda.
        """
        if not isinstance(jit_lambda, JITLambda):
            raise TypeError(
                f"jit_lambda must be of type JITLambda, not {type(jit_lambda)}"
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
                jit_lambda.cpp(), public_arguments.cpp(), evaluation_keys.cpp()
            )
        )
