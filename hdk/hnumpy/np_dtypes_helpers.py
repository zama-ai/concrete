"""File to hold code to manage package and numpy dtypes"""

from copy import deepcopy

import numpy
from numpy.typing import DTypeLike

from ..common.data_types.base import BaseDataType
from ..common.data_types.floats import Float
from ..common.data_types.integers import Integer

NUMPY_TO_HDK_TYPE_MAPPING = {
    numpy.dtype(numpy.int32): Integer(32, is_signed=True),
    numpy.dtype(numpy.int64): Integer(64, is_signed=True),
    numpy.dtype(numpy.uint32): Integer(32, is_signed=False),
    numpy.dtype(numpy.uint64): Integer(64, is_signed=False),
    numpy.dtype(numpy.float32): Float(32),
    numpy.dtype(numpy.float64): Float(64),
}

SUPPORTED_NUMPY_TYPES_SET = NUMPY_TO_HDK_TYPE_MAPPING.keys()


def convert_numpy_dtype_to_common_dtype(numpy_dtype: DTypeLike) -> BaseDataType:
    """Helper function to get the corresponding type from a numpy dtype

    Args:
        numpy_dtype (DTypeLike): Any python object that can be translated to a numpy.dtype

    Raises:
        ValueError: If the numpy_dtype is not supported

    Returns:
        BaseDataType: The corresponding data type corresponding to the input numpy_dtype
    """

    # Normalize numpy_dtype
    normalized_numpy_dtype = numpy.dtype(numpy_dtype)
    corresponding_hdk_dtype = NUMPY_TO_HDK_TYPE_MAPPING.get(normalized_numpy_dtype, None)

    if corresponding_hdk_dtype is None:
        raise ValueError(
            f"Unsupported numpy type: {numpy_dtype} ({normalized_numpy_dtype}), "
            f"supported numpy types: "
            f"{', '.join(sorted(str(dtype) for dtype in SUPPORTED_NUMPY_TYPES_SET))}"
        )

    # deepcopy to avoid having the value from the dict modified
    return deepcopy(corresponding_hdk_dtype)
