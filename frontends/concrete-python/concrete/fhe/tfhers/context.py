"""
Declaration of `tfhers.Bridge` class.
"""

from typing import List, Optional, Union

from concrete.compiler import TfhersExporter, TfhersFheIntDescription

from concrete import fhe

from .dtypes import EncryptionKeyChoice, TFHERSIntegerType


class Bridge:
    """TFHErs Bridge extend a Circuit with TFHErs functionalities."""

    circuit: "fhe.Circuit"
    input_types: Union[List[Optional[TFHERSIntegerType]], Optional[TFHERSIntegerType]]
    output_types: Union[List[Optional[TFHERSIntegerType]], Optional[TFHERSIntegerType]]
    func_name: str

    def __init__(
        self,
        circuit: "fhe.Circuit",
        input_types: Union[List[Optional[TFHERSIntegerType]], Optional[TFHERSIntegerType]],
        output_types: Union[List[Optional[TFHERSIntegerType]], Optional[TFHERSIntegerType]],
        func_name: str,
    ):
        self.circuit = circuit
        self.input_types = input_types
        self.output_types = output_types
        self.func_name = func_name

    def _input_type(self, input_idx: int) -> Optional[TFHERSIntegerType]:
        """Return the type of a certain input.

        Args:
            input_idx (int): the input index to get the type of

        Returns:
            Optional[TFHERSIntegerType]: input type. None means a non-tfhers type
        """
        if isinstance(self.input_types, list):
            return self.input_types[input_idx]
        return self.input_types

    def _output_type(self, output_idx: int) -> Optional[TFHERSIntegerType]:
        """Return the type of a certain output.

        Args:
            output_idx (int): the output index to get the type of

        Returns:
            Optional[TFHERSIntegerType]: output type. None means a non-tfhers type
        """
        if isinstance(self.output_types, list):
            return self.output_types[output_idx]
        return self.output_types

    def _input_keyid(self, input_idx: int) -> int:
        return self.circuit.client.specs.client_parameters.input_keyid_at(input_idx, self.func_name)

    def _input_variance(self, input_idx: int) -> float:
        input_type = self._input_type(input_idx)
        if input_type is None:  # pragma: no cover
            msg = "input at 'input_idx' is not a TFHErs value"
            raise ValueError(msg)
        return input_type.params.encryption_variance()

    @staticmethod
    def _description_from_type(
        tfhers_int_type: TFHERSIntegerType,
    ) -> TfhersFheIntDescription:
        """Construct a TFHErs integer description based on type."""

        bit_width = tfhers_int_type.bit_width
        signed = tfhers_int_type.is_signed
        params = tfhers_int_type.params
        message_modulus = 2**tfhers_int_type.msg_width
        carry_modulus = 2**tfhers_int_type.carry_width
        lwe_size = params.polynomial_size + 1
        n_cts = bit_width // tfhers_int_type.msg_width
        ks_first = params.encryption_key_choice is EncryptionKeyChoice.BIG
        # maximum value using message bits as we don't use carry bits here
        degree = message_modulus - 1
        # this should imply running a PBS on TFHErs side
        noise_level = TfhersFheIntDescription.get_unknown_noise_level()

        return TfhersFheIntDescription.new(
            bit_width,
            signed,
            message_modulus,
            carry_modulus,
            degree,
            lwe_size,
            n_cts,
            noise_level,
            ks_first,
        )

    def import_value(self, buffer: bytes, input_idx: int) -> "fhe.Value":
        """Import a serialized TFHErs integer as a Value.

        Args:
            buffer (bytes): serialized integer
            input_idx (int): the index of the input expecting this value

        Returns:
            fhe.Value: imported value
        """
        input_type = self._input_type(input_idx)
        if input_type is None:  # pragma: no cover
            msg = "input at 'input_idx' is not a TFHErs value"
            raise ValueError(msg)

        fheint_desc = self._description_from_type(input_type)

        bit_width = input_type.bit_width
        signed = input_type.is_signed
        if bit_width == 8:
            if not signed:
                keyid = self._input_keyid(input_idx)
                variance = self._input_variance(input_idx)
                return fhe.Value(
                    TfhersExporter.import_fheuint8(buffer, fheint_desc, keyid, variance)
                )

        msg = (
            f"importing {'signed' if signed else 'unsigned'} integers of {bit_width}bits is not"
            " yet supported"
        )
        raise NotImplementedError(msg)

    def export_value(self, value: "fhe.Value", output_idx: int) -> bytes:
        """Export a value as a serialized TFHErs integer.

        Args:
            value (fhe.Value): value to export
            output_idx (int): the index corresponding to this output

        Returns:
            bytes: serialized fheuint8
        """
        output_type = self._output_type(output_idx)
        if output_type is None:  # pragma: no cover
            msg = "output at 'output_idx' is not a TFHErs value"
            raise ValueError(msg)

        fheint_desc = self._description_from_type(output_type)

        bit_width = output_type.bit_width
        signed = output_type.is_signed
        if bit_width == 8:
            if not signed:
                return TfhersExporter.export_fheuint8(value.inner, fheint_desc)

        msg = (
            f"exporting value to {'signed' if signed else 'unsigned'} integers of {bit_width}bits"
            " is not yet supported"
        )
        raise NotImplementedError(msg)

    def serialize_input_secret_key(self, input_idx: int) -> bytes:
        """Serialize secret key used for a specific input.

        Args:
            input_idx (int): input index corresponding to the key to serialize

        Returns:
            bytes: serialized key
        """
        keyid = self._input_keyid(input_idx)
        input_type = self._input_type(input_idx)
        if input_type is None:  # pragma: no cover
            msg = "input at 'input_idx' is not a TFHErs value"
            raise ValueError(msg)
        glwe_dim = input_type.params.glwe_dimension
        poly_size = input_type.params.polynomial_size
        # pylint: disable=protected-access
        return self.circuit.client.keys._keyset.serialize_lwe_secret_key_as_glwe(  # type: ignore
            keyid, glwe_dim, poly_size
        )
        # pylint: enable=protected-access


def new_bridge(
    circuit: "fhe.Circuit",
    input_types: Union[List[Optional[TFHERSIntegerType]], Optional[TFHERSIntegerType]],
    output_types: Union[List[Optional[TFHERSIntegerType]], Optional[TFHERSIntegerType]],
    func_name: str = "main",
) -> Bridge:
    """Create a TFHErs bridge from a circuit.

    Args:
        circuit (Circuit): compiled circuit
        input_types (Union[List[Optional[TFHERSIntegerType]], Optional[TFHERSIntegerType]]): lists
            should map every input to a type, while a single element is general for all inputs.
            None means a non-tfhers type
        output_types (Union[List[Optional[TFHERSIntegerType]], Optional[TFHERSIntegerType]]): lists
            should map every output to a type, while a single element is general for all outputs.
            None means a non-tfhers type
        func_name (str, optional): name of the function to use. Defaults to "main".

    Returns:
        Bridge: TFHErs bridge
    """
    return Bridge(circuit, input_types, output_types, func_name)
