"""
Declaration of `ClientSpecs` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

from typing import Any

# mypy: disable-error-code=attr-defined
from concrete.compiler import ProgramInfo

# pylint: enable=import-error,no-member,no-name-in-module


class ClientSpecs:
    """
    ClientSpecs class, to create Client objects.
    """

    program_info: ProgramInfo

    def __init__(self, program_info: ProgramInfo):
        self.program_info = program_info

    def __eq__(self, other: Any):  # pragma: no cover
        return self.program_info.serialize() == other.program_info.serialize()

    def serialize(self) -> bytes:
        """
        Serialize client specs into a string representation.

        Returns:
            bytes:
                serialized client specs
        """

        return self.program_info.serialize()

    @staticmethod
    def deserialize(serialized_client_specs: bytes) -> "ClientSpecs":
        """
        Create client specs from its string representation.

        Args:
            serialized_client_specs (bytes):
                client specs to deserialize

        Returns:
            ClientSpecs:
                deserialized client specs
        """

        program_info = ProgramInfo.deserialize(serialized_client_specs)
        return ClientSpecs(program_info)
