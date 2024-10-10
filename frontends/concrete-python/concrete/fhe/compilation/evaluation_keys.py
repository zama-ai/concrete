"""
Declaration of `EvaluationKeys`.
"""

# pylint: disable=import-error,no-member,no-name-in-module
from concrete.compiler import ServerKeyset
from typing_extensions import NamedTuple


class EvaluationKeys(NamedTuple):
    """
    EvaluationKeys required for execution.
    """

    server_keyset: ServerKeyset

    def serialize(self) -> bytes:
        """
        Serialize the evaluation keys.
        """
        return self.server_keyset.serialize()

    @staticmethod
    def deserialize(buffer: bytes) -> "EvaluationKeys":
        """
        Deserialize evaluation keys from bytes.
        """
        return EvaluationKeys(ServerKeyset.deserialize(buffer))
