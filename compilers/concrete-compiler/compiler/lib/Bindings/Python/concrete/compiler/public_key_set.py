#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.


"""KeySet.

Store for the different keys required for an public encryption.
"""


# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    PublicKeySet as _PublicKeySet,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class PublicKeySet(WrapperCpp):
    """KeySet stores the different keys required for a public computation.

    Holds public keys used for encryption.
    """

    def __init__(self, keyset: _PublicKeySet):
        """Wrap the native Cpp object.

        Args:
            keyset (_PublicKeySet): object to wrap

        Raises:
            TypeError: if keyset is not of type _PublicKeySet
        """
        if not isinstance(keyset, _PublicKeySet):
            raise TypeError(f"keyset must be of type _PublicKeySet, not {type(keyset)}")
        super().__init__(keyset)

    def serialize(self) -> bytes:
        """Serialize the KeySet.

        Returns:
            bytes: serialized object
        """
        return self.cpp().serialize()

    @staticmethod
    def deserialize(serialized_key_set: bytes) -> "PublicKeySet":
        """Deserialize KeySet from bytes.

        Args:
            serialized_key_set (bytes): previously serialized KeySet

        Raises:
            TypeError: if serialized_key_set is not of type bytes

        Returns:
            KeySet: deserialized object
        """
        if not isinstance(serialized_key_set, bytes):
            raise TypeError(
                f"serialized_key_set must be of type bytes, not {type(serialized_key_set)}"
            )
        return PublicKeySet.wrap(_PublicKeySet.deserialize(serialized_key_set))
