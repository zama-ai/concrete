"""Import and export TFHErs integers into Concrete."""

from typing import Tuple

# pylint: disable=no-name-in-module,import-error,

from mlir._mlir_libs._concretelang._compiler import (
    import_tfhers_int as _import_tfhers_int,
    export_tfhers_int as _export_tfhers_int,
    TfhersFheIntDescription,
    TransportValue,
)

# pylint: enable=no-name-in-module,import-error


class TfhersExporter:
    """A helper class to import and export TFHErs big integers."""

    @staticmethod
    def export_int(value: TransportValue, info: TfhersFheIntDescription) -> bytes:
        """Convert Concrete value to TFHErs and serialize it.

        Args:

            value (Value): value to export
            info (TfhersFheIntDescription): description of the TFHErs integer to export to

        Raises:
            TypeError: if wrong input types

        Returns:
            bytes: converted and serialized TFHErs integer
        """
        if not isinstance(value, TransportValue):
            raise TypeError(f"value must be of type TransportValue, not {type(value)}")
        if not isinstance(info, TfhersFheIntDescription):
            raise TypeError(
                f"info must be of type TfhersFheIntDescription, not {type(info)}"
            )
        return bytes(_export_tfhers_int(value, info))

    @staticmethod
    def import_int(
        buffer: bytes,
        info: TfhersFheIntDescription,
        keyid: int,
        variance: float,
        shape: Tuple[int, ...],
    ) -> TransportValue:
        """Unserialize and convert from TFHErs to Concrete value.

        Args:
            buffer (bytes): serialized TFHErs integer
            info (TfhersFheIntDescription): description of the TFHErs integer to import
            keyid (int): id of the key used for encryption
            variance (float): variance used for encryption
            shape (Tuple[int, ...]): expected shape

        Raises:
            TypeError: if wrong input types

        Returns:
            TransportValue: unserialized and converted value
        """
        if not isinstance(buffer, bytes):
            raise TypeError(f"buffer must be of type bytes, not {type(buffer)}")
        if not isinstance(info, TfhersFheIntDescription):
            raise TypeError(
                f"info must be of type TfhersFheIntDescription, not {type(info)}"
            )
        if not isinstance(keyid, int):
            raise TypeError(f"keyid must be of type int, not {type(keyid)}")
        if not isinstance(variance, float):
            raise TypeError(f"variance must be of type float, not {type(variance)}")
        if not isinstance(shape, tuple):
            raise TypeError(f"shape must be of type tuple(int, ...), not {type(shape)}")
        return _import_tfhers_int(buffer, info, keyid, variance, shape)
