#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

"""CompilationContext.

CompilationContext holds the MLIR Context supposed to be used during IR generation.
"""

# pylint: disable=no-name-in-module,import-error,too-many-instance-attributes,protected-access
from mlir._mlir_libs._concretelang._compiler import (
    CompilationContext as _CompilationContext,
)
from mlir.ir import Context as MlirContext


class CompilationContext(_CompilationContext):
    """
    Compilation context.
    """

    @staticmethod
    def new() -> "CompilationContext":
        """
        Creates a new CompilationContext.
        """
        return CompilationContext()

    def mlir_context(self) -> MlirContext:
        """
        Returns the associated mlir context.
        """
        return MlirContext._CAPICreate(_CompilationContext.mlir_context(self))
