#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

"""CompilationOptions."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    CompilationOptions as _CompilationOptions,
)
from .wrapper import WrapperCpp

# pylint: enable=no-name-in-module,import-error


class CompilationOptions(WrapperCpp):
    """CompilationOptions holds different flags and options of the compilation process.

    It controls different parallelization flags, diagnostic verification, and also the name of entrypoint
    function.
    """

    def __init__(self, compilation_options: _CompilationOptions):
        """Wrap the native Cpp object.

        Args:
            compilation_options (_CompilationOptions): object to wrap

        Raises:
            TypeError: if compilation_options is not of type _CompilationOptions
        """
        if not isinstance(compilation_options, _CompilationOptions):
            raise TypeError(
                f"_compilation_options must be of type _CompilationOptions, not {type(compilation_options)}"
            )
        super().__init__(compilation_options)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(function_name="main") -> "CompilationOptions":
        """Build a CompilationOptions.

        Args:
            function_name (str, optional): name of the entrypoint function. Defaults to "main".

        Raises:
            TypeError: if function_name is not an str

        Returns:
            CompilationOptions
        """
        if not isinstance(function_name, str):
            raise TypeError(
                f"function_name must be of type str not {type(function_name)}"
            )
        return CompilationOptions.wrap(_CompilationOptions(function_name))

    # pylint: enable=arguments-differ

    def set_auto_parallelize(self, auto_parallelize: bool):
        """Set option for auto parallelization.

        Args:
            auto_parallelize (bool): whether to turn it on or off

        Raises:
            TypeError: if the value to set is not boolean
        """
        if not isinstance(auto_parallelize, bool):
            raise TypeError("can't set the option to a non-boolean value")
        self.cpp().set_auto_parallelize(auto_parallelize)

    def set_loop_parallelize(self, loop_parallelize: bool):
        """Set option for loop parallelization.

        Args:
            loop_parallelize (bool): whether to turn it on or off

        Raises:
            TypeError: if the value to set is not boolean
        """
        if not isinstance(loop_parallelize, bool):
            raise TypeError("can't set the option to a non-boolean value")
        self.cpp().set_loop_parallelize(loop_parallelize)

    def set_verify_diagnostics(self, verify_diagnostics: bool):
        """Set option for diagnostics verification.

        Args:
            verify_diagnostics (bool): whether to turn it on or off

        Raises:
            TypeError: if the value to set is not boolean
        """
        if not isinstance(verify_diagnostics, bool):
            raise TypeError("can't set the option to a non-boolean value")
        self.cpp().set_verify_diagnostics(verify_diagnostics)

    def set_dataflow_parallelize(self, dataflow_parallelize: bool):
        """Set option for dataflow parallelization.

        Args:
            dataflow_parallelize (bool): whether to turn it on or off

        Raises:
            TypeError: if the value to set is not boolean
        """
        if not isinstance(dataflow_parallelize, bool):
            raise TypeError("can't set the option to a non-boolean value")
        self.cpp().set_dataflow_parallelize(dataflow_parallelize)

    def set_funcname(self, funcname: str):
        """Set entrypoint function name.

        Args:
            funcname (str): name of the entrypoint function

        Raises:
            TypeError: if the value to set is not str
        """
        if not isinstance(funcname, str):
            raise TypeError("can't set the option to a non-str value")
        self.cpp().set_funcname(funcname)
