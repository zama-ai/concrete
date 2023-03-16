"""
Declaration of `ClientSpecs` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

import json
from typing import List

# mypy: disable-error-code=attr-defined
from concrete.compiler import ClientParameters, PublicArguments, PublicResult

# pylint: enable=import-error,no-member,no-name-in-module


class ClientSpecs:
    """
    ClientSpecs class, to create Client objects.
    """

    input_signs: List[bool]
    client_parameters: ClientParameters
    output_signs: List[bool]

    def __init__(
        self,
        input_signs: List[bool],
        client_parameters: ClientParameters,
        output_signs: List[bool],
    ):
        self.input_signs = input_signs
        self.client_parameters = client_parameters
        self.output_signs = output_signs

    def serialize(self) -> str:
        """
        Serialize client specs into a string representation.

        Returns:
            str:
                string representation of the client specs
        """

        client_parameters_json = json.loads(self.client_parameters.serialize())
        return json.dumps(
            {
                "input_signs": self.input_signs,
                "client_parameters": client_parameters_json,
                "output_signs": self.output_signs,
            }
        )

    @staticmethod
    def deserialize(serialized_client_specs: str) -> "ClientSpecs":
        """
        Create client specs from its string representation.

        Args:
            serialized_client_specs (str):
                client specs to deserialize

        Returns:
            ClientSpecs:
                deserialized client specs
        """

        raw_specs = json.loads(serialized_client_specs)

        client_parameters_bytes = json.dumps(raw_specs["client_parameters"]).encode("utf-8")
        client_parameters = ClientParameters.deserialize(client_parameters_bytes)

        return ClientSpecs(raw_specs["input_signs"], client_parameters, raw_specs["output_signs"])

    def serialize_public_args(self, args: PublicArguments) -> bytes:
        """
        Serialize public arguments to bytes.

        Args:
            args (PublicArguments):
                public arguments to serialize

        Returns:
            bytes:
                serialized public arguments
        """

        return args.serialize()

    def deserialize_public_args(self, serialized_args: bytes) -> PublicArguments:
        """
        Unserialize public arguments from bytes.

        Args:
            serialized_args (bytes):
                serialized public arguments

        Returns:
            PublicArguments:
                deserialized public arguments
        """

        return PublicArguments.deserialize(self.client_parameters, serialized_args)

    def serialize_public_result(self, result: PublicResult) -> bytes:
        """
        Serialize public result to bytes.

        Args:
            result (PublicResult):
                public result to serialize

        Returns:
            bytes:
                serialized public result
        """

        return result.serialize()

    def deserialize_public_result(self, serialized_result: bytes) -> PublicResult:
        """
        Unserialize public result from bytes.

        Args:
            serialized_result (bytes):
                serialized public result

        Returns:
            PublicResult:
                deserialized public result
        """

        return PublicResult.deserialize(self.client_parameters, serialized_result)
