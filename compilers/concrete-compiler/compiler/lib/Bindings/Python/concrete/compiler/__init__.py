#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

"""Compiler submodule."""
import atexit
import json
from typing import Union
import jsonpickle

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    LweSecretKeyParam,
    BootstrapKeyParam,
    KeyswitchKeyParam,
    PackingKeyswitchKeyParam,
    ProgramInfo,
    CompilationOptions,
    LweSecretKey,
    KeysetCache,
    ServerKeyset,
    Keyset,
    Compiler,
    TfhersFheIntDescription,
    TransportValue,
    Value,
    ServerProgram,
    ServerCircuit,
    ClientProgram,
    ClientCircuit,
    Backend,
    KeyType,
    OptimizerMultiParameterStrategy,
    OptimizerStrategy,
    PrimitiveOperation,
    Library,
    ProgramCompilationFeedback,
    CircuitCompilationFeedback,
    KeysetRestriction,
    RangeRestriction,
    terminate_df_parallelization as _terminate_df_parallelization,
    init_df_parallelization as _init_df_parallelization,
    check_gpu_runtime_enabled as _check_gpu_runtime_enabled,
    check_cuda_device_available as _check_cuda_device_available,
    round_trip as _round_trip,
    set_llvm_debug_flag,
    set_compiler_logging,
)

# pylint: enable=no-name-in-module,import-error

from .utils import lookup_runtime_lib
from .compilation_feedback import MoreCircuitCompilationFeedback
from .compilation_context import CompilationContext

from .tfhers_int import TfhersExporter

Parameter = Union[
    LweSecretKeyParam, BootstrapKeyParam, KeyswitchKeyParam, PackingKeyswitchKeyParam
]


def init_dfr():
    """Initialize dataflow parallelization.

    It is not always required to initialize the dataflow runtime as it can be implicitely done
    during compilation. However, it is required in case no compilation has previously been done
    and the runtime is needed"""
    _init_df_parallelization()


def check_gpu_enabled() -> bool:
    """Check whether the compiler and runtime support GPU offloading.

    GPU offloading is not always available, in particular in non-GPU wheels."""
    return _check_gpu_runtime_enabled()


def check_gpu_available() -> bool:
    """Check whether a CUDA device is available and online."""
    return _check_cuda_device_available()


# Cleanly terminate the dataflow runtime if it has been initialized
# (does nothing otherwise)
atexit.register(_terminate_df_parallelization)


def round_trip(mlir_str: str) -> str:
    """Parse the MLIR input, then return it back.

    Useful to check the validity of an MLIR representation

    Args:
        mlir_str (str): textual representation of an MLIR code

    Raises:
        TypeError: if mlir_str is not of type str

    Returns:
        str: textual representation of the MLIR code after parsing
    """
    if not isinstance(mlir_str, str):
        raise TypeError(f"mlir_str must be of type str, not {type(mlir_str)}")
    return _round_trip(mlir_str)


@jsonpickle.handlers.register(RangeRestriction)
class RangeRestrictionHandler(jsonpickle.handlers.BaseHandler):
    """Handler to serialize and deserialize range restrictions"""

    def flatten(self, obj, data):
        data["serialized"] = json.loads(obj.to_json())
        return data

    def restore(self, obj):
        return RangeRestriction.from_json(json.dumps(obj["serialized"]))


@jsonpickle.handlers.register(KeysetRestriction)
class KeysetRestrictionHandler(jsonpickle.handlers.BaseHandler):
    """Handler to serialize and deserialize keyset restrictions"""

    def flatten(self, obj, data):
        data["serialized"] = json.loads(obj.to_json())
        return data

    def restore(self, obj):
        return KeysetRestriction.from_json(json.dumps(obj["serialized"]))
