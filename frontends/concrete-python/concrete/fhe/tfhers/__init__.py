"""
tfhers module to represent, and compute on tfhers integer values.
"""

from .dtypes import (
    TFHERSIntegerType,
    int8,
    int8_2_2,
    int16,
    int16_2_2,
    uint8,
    uint8_2_2,
    uint16,
    uint16_2_2,
)
from .tracing import from_native, to_native
from .values import (
    TFHERSInteger,
    int8_2_2_value,
    int16_2_2_value,
    uint8_2_2_value,
    uint16_2_2_value,
)
