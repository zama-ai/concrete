#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""LibraryCompilationResult."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    LibraryCompilationResult as _LibraryCompilationResult,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class LibraryCompilationResult(WrapperCpp):
    """LibraryCompilationResult holds the result of the library compilation."""

    def __init__(self, library_compilation_result: _LibraryCompilationResult):
        """Wrap the native Cpp object.

        Args:
            library_compilation_result (_LibraryCompilationResult): object to wrap

        Raises:
            TypeError: if library_compilation_result is not of type _LibraryCompilationResult
        """
        if not isinstance(library_compilation_result, _LibraryCompilationResult):
            raise TypeError(
                f"library_compilation_result must be of type _LibraryCompilationResult, not "
                f"{type(library_compilation_result)}"
            )
        super().__init__(library_compilation_result)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(output_dir_path: str, func_name: str) -> "LibraryCompilationResult":
        """Build a LibraryCompilationResult at output_dir_path, with func_name as entrypoint.

        Args:
            output_dir_path (str): path to the compilation artifacts
            func_name (str): entrypoint function name

        Raises:
            TypeError: if output_dir_path is not of type str
            TypeError: if func_name is not of type str

        Returns:
            LibraryCompilationResult
        """
        if not isinstance(output_dir_path, str):
            raise TypeError(
                f"output_dir_path must be of type str, not {type(output_dir_path)}"
            )
        if not isinstance(func_name, str):
            raise TypeError(f"func_name must be of type str, not {type(func_name)}")
        return LibraryCompilationResult.wrap(
            _LibraryCompilationResult(output_dir_path, func_name)
        )

    # pylint: enable=arguments-differ
