#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""Compilation feedback."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    CompilationFeedback as _CompilationFeedback,
)

# pylint: enable=no-name-in-module,import-error

from .wrapper import WrapperCpp


class CompilationFeedback(WrapperCpp):
    """CompilationFeedback is a set of hint computed by the compiler engine."""

    def __init__(self, compilation_feedback: _CompilationFeedback):
        """Wrap the native Cpp object.

        Args:
            compilation_feeback (_CompilationFeedback): object to wrap

        Raises:
            TypeError: if compilation_feedback is not of type _CompilationFeedback
        """
        if not isinstance(compilation_feedback, _CompilationFeedback):
            raise TypeError(
                f"compilation_feedback must be of type _CompilationFeedback, not {type(compilation_feedback)}"
            )

        self.complexity = compilation_feedback.complexity
        self.p_error = compilation_feedback.p_error
        self.global_p_error = compilation_feedback.global_p_error
        self.total_secret_keys_size = compilation_feedback.total_secret_keys_size
        self.total_bootstrap_keys_size = compilation_feedback.total_bootstrap_keys_size
        self.total_keyswitch_keys_size = compilation_feedback.total_keyswitch_keys_size
        self.total_inputs_size = compilation_feedback.total_inputs_size
        self.total_output_size = compilation_feedback.total_output_size
        self.crt_decompositions_of_outputs = (
            compilation_feedback.crt_decompositions_of_outputs
        )

        super().__init__(compilation_feedback)
