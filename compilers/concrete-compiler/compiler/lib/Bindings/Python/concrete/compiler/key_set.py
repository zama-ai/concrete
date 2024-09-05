#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.


"""KeySet.

Store for the different keys required for an encrypted computation.
"""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    KeySet as _KeySet,
)

# pylint: enable=no-name-in-module,import-error
from .lwe_secret_key import LweSecretKey
from .wrapper import WrapperCpp
from .evaluation_keys import EvaluationKeys


class KeySet(WrapperCpp):
    """KeySet stores the different keys required for an encrypted computation.

    Holds private keys (secret key) used for encryption/decryption, and public keys used for computation.
    """

    def __init__(self, keyset: _KeySet):
        """Wrap the native Cpp object.

        Args:
            keyset (_KeySet): object to wrap

        Raises:
            TypeError: if keyset is not of type _KeySet
        """
        if not isinstance(keyset, _KeySet):
            raise TypeError(f"keyset must be of type _KeySet, not {type(keyset)}")
        super().__init__(keyset)

    def serialize(self) -> bytes:
        """Serialize the KeySet.

        Returns:
            bytes: serialized object
        """
        return self.cpp().serialize()

    @staticmethod
    def deserialize(serialized_key_set: bytes) -> "KeySet":
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
        return KeySet.wrap(_KeySet.deserialize(serialized_key_set))

    def get_lwe_secret_key(self, keyid: int) -> LweSecretKey:
        """Get a specific LweSecretKey.

        Args:
            keyid (int): id of the key to get

        Raises:
            TypeError: if wrong types for input arguments

        Returns:
            bytes: LweSecretKey
        """
        if not isinstance(keyid, int):
            raise TypeError(f"keyid must be of type int, not {type(keyid)}")
        return LweSecretKey.wrap(self.cpp().get_lwe_secret_key(keyid))

    def get_evaluation_keys(self) -> EvaluationKeys:
        """
        Get evaluation keys for execution.

        Returns:
            EvaluationKeys:
                evaluation keys for execution
        """
        return EvaluationKeys(self.cpp().get_evaluation_keys())
