#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""CompilationOptions."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    CompilationOptions as _CompilationOptions,
    OptimizerStrategy as _OptimizerStrategy,
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

    def set_optimize_concrete(self, optimize: bool):
        """Set flag to enable/disable optimization of concrete intermediate representation.

        Args:
            optimize (bool): whether to turn it on or off

        Raises:
            TypeError: if the value to set is not boolean
        """
        if not isinstance(optimize, bool):
            raise TypeError("can't set the option to a non-boolean value")
        self.cpp().set_optimize_concrete(optimize)

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

    def set_p_error(self, p_error: float):
        """Set error probability for shared by each pbs.

        Args:
            p_error (float): probability of error for each lut

        Raises:
            TypeError: if the value to set is not float
            ValueError: if the value to set is not in interval ]0; 1]
        """
        if not isinstance(p_error, float):
            raise TypeError("can't set p_error to a non-float value")
        if p_error == 0.0:
            raise ValueError("p_error cannot be 0")
        if not 0.0 <= p_error <= 1.0:
            raise ValueError("p_error should be a probability in ]0; 1]")
        self.cpp().set_p_error(p_error)

    def set_display_optimizer_choice(self, display: bool):
        """Set display flag of optimizer choices.

        Args:
            display (bool): if true the compiler display optimizer choices

        Raises:
            TypeError: if the value is not a bool
        """
        if not isinstance(display, bool):
            raise TypeError("display should be a bool")
        self.cpp().set_display_optimizer_choice(display)

    def set_optimizer_strategy(self, strategy: _OptimizerStrategy):
        """Set the strategy of the optimizer.

        Args:
            strategy (OptimizerStrategy): Use the specified optmizer strategy.

        Raises:
            TypeError: if the value is not a bool
        """
        if not isinstance(strategy, _OptimizerStrategy):
            raise TypeError("enable should be a bool")
        self.cpp().set_optimizer_strategy(strategy)

    def set_global_p_error(self, global_p_error: float):
        """Set global error probability for the full circuit.

        Args:
            global_p_error (float): probability of error for the full circuit

        Raises:
            TypeError: if the value to set is not float
            ValueError: if the value to set is not in interval ]0; 1]
        """
        if not isinstance(global_p_error, float):
            raise TypeError("can't set global_p_error to a non-float value")
        if global_p_error == 0.0:
            raise ValueError("global_p_error cannot be 0")
        if not 0.0 <= global_p_error <= 1.0:
            raise ValueError("global_p_error be a probability in ]0; 1]")
        self.cpp().set_global_p_error(global_p_error)

    def set_security_level(self, security_level: int):
        """Set security level.

        Args:
            security_level (int): the target number of bits of security to compile the circuit

        Raises:
            TypeError: if the value to set is not int
            ValueError: if the value to set is not in interval ]0; 1]
        """
        if not isinstance(security_level, int):
            raise TypeError("can't set security_level to a non-int value")
        self.cpp().set_security_level(security_level)
