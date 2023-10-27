#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""CompilationContext.

CompilationContext holds the MLIR Context supposed to be used during IR generation.
"""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    CompilationContext as _CompilationContext,
)
from mlir.ir import Context as MlirContext

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class CompilationContext(WrapperCpp):
    """Support class for compilation context.

    CompilationContext is meant to outlive mlir_context().
    Do not use the mlir_context after deleting the CompilationContext.
    """

    def __init__(self, compilation_context: _CompilationContext):
        """Wrap the native Cpp object.

        Args:
            compilation_context (_CompilationContext): object to wrap

        Raises:
            TypeError: if compilation_context is not of type _CompilationContext
        """
        if not isinstance(compilation_context, _CompilationContext):
            raise TypeError(
                f"compilation_context must be of type _CompilationContext, not "
                f"{type(compilation_context)}"
            )
        super().__init__(compilation_context)

    # pylint: disable=arguments-differ
    @staticmethod
    def new() -> "CompilationContext":
        """Build a CompilationContext.

        Returns:
            CompilationContext
        """
        return CompilationContext.wrap(_CompilationContext())

    def mlir_context(
        self,
    ) -> MlirContext:
        """
        Get the MLIR context used by the compilation context.

        The Compilation Context should outlive the mlir_context.

        Returns:
            MlirContext: MLIR context of the compilation context
        """
        # pylint: disable=protected-access
        return MlirContext._CAPICreate(self.cpp().mlir_context())
        # pylint: enable=protected-access
