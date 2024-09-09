"""
Declaration of `EncryptionStatus` class.
"""

# pylint: disable=import-error,no-name-in-module

from enum import Enum, unique


@unique
class EncryptionStatus(str, Enum):
    """
    EncryptionStatus enum, to represent encryption status of parameters.
    """

    CLEAR = "clear"
    ENCRYPTED = "encrypted"
