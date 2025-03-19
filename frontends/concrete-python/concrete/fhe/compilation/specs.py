"""
Declaration of `ClientSpecs` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module
import json
from typing import Any, Optional

# mypy: disable-error-code=attr-defined
from concrete.compiler import ProgramInfo

# pylint: enable=import-error,no-member,no-name-in-module
from concrete import fhe

TFHERS_SPECS_KEY = "tfhers_specs"


class ClientSpecs:
    """
    ClientSpecs class, to create Client objects.
    """

    program_info: ProgramInfo
    tfhers_specs: Optional["fhe.tfhers.TFHERSClientSpecs"]

    def __init__(
        self,
        program_info: ProgramInfo,
        tfhers_specs: Optional["fhe.tfhers.TFHERSClientSpecs"] = None,
    ):
        self.program_info = program_info
        self.tfhers_specs = tfhers_specs

    def __eq__(self, other: Any):  # pragma: no cover
        return (
            self.program_info.serialize() == other.program_info.serialize()
            and self.tfhers_specs == other.tfhers_specs
        )

    def serialize(self) -> bytes:
        """
        Serialize client specs into bytes.

        Returns:
            bytes:
                serialized client specs
        """
        program_info = json.loads(self.program_info.serialize())
        if self.tfhers_specs is not None:
            program_info[TFHERS_SPECS_KEY] = self.tfhers_specs.to_dict()
        return json.dumps(program_info).encode("utf-8")

    @staticmethod
    def deserialize(serialized_client_specs: bytes) -> "ClientSpecs":
        """
        Create client specs from bytes.

        Args:
            serialized_client_specs (bytes):
                client specs to deserialize

        Returns:
            ClientSpecs:
                deserialized client specs
        """
        program_info_dict = json.loads(serialized_client_specs)
        tfhers_specs_dict = program_info_dict.get(TFHERS_SPECS_KEY, None)

        if tfhers_specs_dict is not None:
            tfhers_specs = fhe.tfhers.TFHERSClientSpecs.from_dict(tfhers_specs_dict)
            del program_info_dict[TFHERS_SPECS_KEY]
        else:
            tfhers_specs = None

        program_info = ProgramInfo.deserialize(json.dumps(program_info_dict).encode("utf-8"))
        return ClientSpecs(program_info, tfhers_specs)
