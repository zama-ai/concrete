"""Test file for values related code."""

from copy import deepcopy
from functools import partial
from typing import Callable, Optional, Tuple, Union

import pytest

from concrete.common.data_types.base import BaseDataType
from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.common.values import ClearTensor, EncryptedTensor, TensorValue


class DummyDtype(BaseDataType):
    """Dummy Helper Dtype"""

    def __eq__(self, o: object) -> bool:
        return isinstance(o, self.__class__)


@pytest.mark.parametrize(
    "tensor_constructor,expected_is_encrypted",
    [
        (ClearTensor, False),
        (partial(TensorValue, is_encrypted=False), False),
        (EncryptedTensor, True),
        (partial(TensorValue, is_encrypted=True), True),
    ],
)
@pytest.mark.parametrize(
    "shape,expected_shape,expected_ndim,expected_size",
    [
        (None, (), 0, 1),
        ((), (), 0, 1),
        ((3, 256, 256), (3, 256, 256), 3, 196_608),
        ((1920, 1080, 3), (1920, 1080, 3), 3, 6_220_800),
    ],
)
@pytest.mark.parametrize(
    "data_type",
    [
        Integer(7, False),
        Integer(32, True),
        Integer(32, False),
        Integer(64, True),
        Integer(64, False),
        Float(32),
        Float(64),
    ],
)
def test_tensor_value(
    tensor_constructor: Callable[..., TensorValue],
    expected_is_encrypted: bool,
    shape: Optional[Tuple[int, ...]],
    expected_shape: Tuple[int, ...],
    expected_ndim: int,
    expected_size: int,
    data_type: Union[Integer, Float],
):
    """Test function for TensorValue"""

    tensor_value = tensor_constructor(dtype=data_type, shape=shape)

    assert expected_is_encrypted == tensor_value.is_encrypted
    assert expected_shape == tensor_value.shape
    assert expected_ndim == tensor_value.ndim
    assert expected_size == tensor_value.size

    assert data_type == tensor_value.dtype

    other_tensor = deepcopy(tensor_value)

    assert other_tensor == tensor_value

    other_tensor_value = deepcopy(other_tensor)
    other_tensor_value.dtype = DummyDtype()
    assert other_tensor_value != tensor_value

    other_shape = tuple(val + 1 for val in shape) if shape is not None else ()
    other_shape += (2,)
    other_tensor_value = tensor_constructor(dtype=data_type, shape=other_shape)

    assert other_tensor_value.shape != tensor_value.shape
    assert other_tensor_value.ndim != tensor_value.ndim
    assert other_tensor_value.size != tensor_value.size
    assert other_tensor_value != tensor_value
