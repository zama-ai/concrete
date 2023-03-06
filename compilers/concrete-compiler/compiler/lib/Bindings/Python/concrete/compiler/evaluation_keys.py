#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""EvaluationKeys."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    EvaluationKeys as _EvaluationKeys,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class EvaluationKeys(WrapperCpp):
    """
    EvaluationKeys required for execution.
    """

    def __init__(self, evaluation_keys: _EvaluationKeys):
        """Wrap the native Cpp object.

        Args:
            evaluation_keys (_EvaluationKeys): object to wrap

        Raises:
            TypeError: if evaluation_keys is not of type _EvaluationKeys
        """
        if not isinstance(evaluation_keys, _EvaluationKeys):
            raise TypeError(
                f"evaluation_keys must be of type _EvaluationKeys, not {type(evaluation_keys)}"
            )
        super().__init__(evaluation_keys)

    def serialize(self) -> bytes:
        """Serialize the EvaluationKeys.

        Returns:
            bytes: serialized object
        """
        return self.cpp().serialize()

    @staticmethod
    def deserialize(serialized_evaluation_keys: bytes) -> "EvaluationKeys":
        """Unserialize EvaluationKeys from bytes.

        Args:
            serialized_evaluation_keys (bytes): previously serialized EvaluationKeys

        Raises:
            TypeError: if serialized_evaluation_keys is not of type bytes

        Returns:
            EvaluationKeys: deserialized object
        """
        if not isinstance(serialized_evaluation_keys, bytes):
            raise TypeError(
                f"serialized_evaluation_keys must be of type bytes, "
                f"not {type(serialized_evaluation_keys)}"
            )
        return EvaluationKeys.wrap(
            _EvaluationKeys.deserialize(serialized_evaluation_keys)
        )
