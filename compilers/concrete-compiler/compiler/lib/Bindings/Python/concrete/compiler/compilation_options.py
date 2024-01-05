#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""CompilationOptions."""

from typing import List

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    CompilationOptions as _CompilationOptions,
    OptimizerStrategy as _OptimizerStrategy,
    Encoding,
    Backend as _Backend,
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
    def new(function_name="main", backend=_Backend.CPU) -> "CompilationOptions":
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
        if not isinstance(backend, _Backend):
            raise TypeError(
                f"backend must be of type Backend not {type(function_name)}"
            )
        return CompilationOptions.wrap(_CompilationOptions(function_name, backend))

    # pylint: enable=arguments-differ

    def set_composable(self, composable: bool):
        """Set option for composition.

        Args:
            composable (bool): whether to turn it on or off

        Raises:
            TypeError: if the value to set is not boolean
        """
        if not isinstance(composable, bool):
            raise TypeError("can't set the option to a non-boolean value")
        self.cpp().set_composable(composable)

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

    def set_compress_evaluation_keys(self, compress_evaluation_keys: bool):
        """Set option for compression of evaluation keys.

        Args:
            compress_evaluation_keys (bool): whether to turn it on or off

        Raises:
            TypeError: if the value to set is not boolean
        """
        if not isinstance(compress_evaluation_keys, bool):
            raise TypeError("can't set the option to a non-boolean value")
        self.cpp().set_compress_evaluation_keys(compress_evaluation_keys)

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

    def set_v0_parameter(
        self,
        glwe_dim: int,
        log_poly_size: int,
        n_small: int,
        br_level: int,
        br_log_base: int,
        ks_level: int,
        ks_log_base: int,
    ):
        """Set the basic V0 parameters.

        Args:
            glwe_dim (int): GLWE dimension
            log_poly_size (int): log of polynomial size
            n_small (int): n
            br_level (int): bootstrap level
            br_log_base (int): bootstrap base log
            ks_level (int): keyswitch level
            ks_log_base (int): keyswitch base log

        Raises:
            TypeError: if parameters are not of type int
        """
        if not isinstance(glwe_dim, int):
            raise TypeError("glwe_dim need to be an integer")
        if not isinstance(log_poly_size, int):
            raise TypeError("log_poly_size need to be an integer")
        if not isinstance(n_small, int):
            raise TypeError("n_small need to be an integer")
        if not isinstance(br_level, int):
            raise TypeError("br_level need to be an integer")
        if not isinstance(br_log_base, int):
            raise TypeError("br_log_base need to be an integer")
        if not isinstance(ks_level, int):
            raise TypeError("ks_level need to be an integer")
        if not isinstance(ks_log_base, int):
            raise TypeError("ks_log_base need to be an integer")
        self.cpp().set_v0_parameter(
            glwe_dim,
            log_poly_size,
            n_small,
            br_level,
            br_log_base,
            ks_level,
            ks_log_base,
        )

    # pylint: disable=too-many-arguments,too-many-branches

    def set_all_v0_parameter(
        self,
        glwe_dim: int,
        log_poly_size: int,
        n_small: int,
        br_level: int,
        br_log_base: int,
        ks_level: int,
        ks_log_base: int,
        crt_decomp: List[int],
        cbs_level: int,
        cbs_log_base: int,
        pks_level: int,
        pks_log_base: int,
        pks_input_lwe_dim: int,
        pks_output_poly_size: int,
    ):
        """Set all the V0 parameters.

        Args:
            glwe_dim (int): GLWE dimension
            log_poly_size (int): log of polynomial size
            n_small (int): n
            br_level (int): bootstrap level
            br_log_base (int): bootstrap base log
            ks_level (int): keyswitch level
            ks_log_base (int): keyswitch base log
            crt_decomp (List[int]): CRT decomposition vector
            cbs_level (int): circuit bootstrap level
            cbs_log_base (int): circuit bootstrap base log
            pks_level (int): packing keyswitch level
            pks_log_base (int): packing keyswitch base log
            pks_input_lwe_dim (int): packing keyswitch input LWE dimension
            pks_output_poly_size (int): packing keyswitch output polynomial size

        Raises:
            TypeError: if parameters are not of type int
        """
        if not isinstance(glwe_dim, int):
            raise TypeError("glwe_dim need to be an integer")
        if not isinstance(log_poly_size, int):
            raise TypeError("log_poly_size need to be an integer")
        if not isinstance(n_small, int):
            raise TypeError("n_small need to be an integer")
        if not isinstance(br_level, int):
            raise TypeError("br_level need to be an integer")
        if not isinstance(br_log_base, int):
            raise TypeError("br_log_base need to be an integer")
        if not isinstance(ks_level, int):
            raise TypeError("ks_level need to be an integer")
        if not isinstance(ks_log_base, int):
            raise TypeError("ks_log_base need to be an integer")
        if not isinstance(crt_decomp, list):
            raise TypeError("crt_decomp need to be a list of integers")
        if not isinstance(cbs_level, int):
            raise TypeError("cbs_level need to be an integer")
        if not isinstance(cbs_log_base, int):
            raise TypeError("cbs_log_base need to be an integer")
        if not isinstance(pks_level, int):
            raise TypeError("pks_level need to be an integer")
        if not isinstance(pks_log_base, int):
            raise TypeError("pks_log_base need to be an integer")
        if not isinstance(pks_input_lwe_dim, int):
            raise TypeError("pks_input_lwe_dim need to be an integer")
        if not isinstance(pks_output_poly_size, int):
            raise TypeError("pks_output_poly_size need to be an integer")
        self.cpp().set_v0_parameter(
            glwe_dim,
            log_poly_size,
            n_small,
            br_level,
            br_log_base,
            ks_level,
            ks_log_base,
            crt_decomp,
            cbs_level,
            cbs_log_base,
            pks_level,
            pks_log_base,
            pks_input_lwe_dim,
            pks_output_poly_size,
        )

    # pylint: enable=too-many-arguments,too-many-branches

    def force_encoding(self, encoding: Encoding):
        """Force the compiler to use a specific encoding.

        Args:
            encoding (Encoding): the encoding to force the compiler to use

        Raises:
            TypeError: if encoding is not of type Encoding
        """
        if not isinstance(encoding, Encoding):
            raise TypeError("encoding need to be of type Encoding")
        self.cpp().force_encoding(encoding)

    def simulation(self, simulate: bool):
        """Enable or disable simulation.

        Args:
            simulate (bool): flag to enable or disable simulation

        Raises:
            TypeError: if the value to set is not bool
        """
        if not isinstance(simulate, bool):
            raise TypeError("need to pass a boolean value")
        self.cpp().simulation(simulate)

    def set_emit_gpu_ops(self, emit_gpu_ops: bool):
        """Set flag that allows gpu ops to be emitted.

        Args:
            emit_gpu_ops (bool): whether to emit gpu ops.

        Raises:
            TypeError: if the value to set is not bool
        """
        if not isinstance(emit_gpu_ops, bool):
            raise TypeError("emit_gpu_ops must be boolean")
        self.cpp().set_emit_gpu_ops(emit_gpu_ops)

    def set_batch_tfhe_ops(self, batch_tfhe_ops: bool):
        """Set flag that triggers the batching of scalar TFHE operations.

        Args:
            batch_tfhe_ops (bool): whether to batch tfhe ops.

        Raises:
            TypeError: if the value to set is not bool
        """
        if not isinstance(batch_tfhe_ops, bool):
            raise TypeError("batch_tfhe_ops must be boolean")
        self.cpp().set_batch_tfhe_ops(batch_tfhe_ops)
