"""Test file for data types helpers"""
import pytest

from concrete.common.data_types.base import BaseDataType
from concrete.common.data_types.dtypes_helpers import (
    broadcast_shapes,
    find_type_to_hold_both_lossy,
    mix_values_determine_holding_dtype,
    value_is_encrypted_scalar_integer,
    value_is_encrypted_scalar_unsigned_integer,
)
from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.common.values import (
    BaseValue,
    ClearScalar,
    ClearTensor,
    EncryptedScalar,
    EncryptedTensor,
)


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            ClearScalar(Integer(8, is_signed=False)),
            False,
            id="ClearScalar 8 bits unsigned Integer",
        ),
        pytest.param(
            EncryptedScalar(Integer(8, is_signed=True)),
            True,
            id="EncryptedScalar 8 bits signed Integer",
        ),
    ],
)
def test_value_is_encrypted_integer(value: BaseValue, expected_result: bool):
    """Test value_is_encrypted_integer helper"""
    assert value_is_encrypted_scalar_integer(value) == expected_result


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            ClearScalar(Integer(8, is_signed=False)),
            False,
            id="ClearScalar 8 bits unsigned Integer",
        ),
        pytest.param(
            EncryptedScalar(Integer(8, is_signed=True)),
            False,
            id="EncryptedScalar 8 bits signed Integer",
        ),
        pytest.param(
            EncryptedScalar(Integer(8, is_signed=False)),
            True,
            id="EncryptedScalar 8 bits unsigned Integer",
        ),
    ],
)
def test_value_is_encrypted_unsigned_integer(value: BaseValue, expected_result: bool):
    """Test value_is_encrypted_unsigned_integer helper"""
    assert value_is_encrypted_scalar_unsigned_integer(value) == expected_result


class UnsupportedDataType(BaseDataType):
    """Test helper class to represent an UnsupportedDataType"""

    def __eq__(self, o: object) -> bool:
        return isinstance(o, self.__class__)


@pytest.mark.parametrize(
    "dtype1,dtype2,expected_mixed_dtype",
    [
        pytest.param(Integer(6, True), Integer(6, True), Integer(6, True), id="int6, int6, int6"),
        pytest.param(
            Integer(6, False), Integer(6, False), Integer(6, False), id="uint6, uint6, uint6"
        ),
        pytest.param(Integer(6, True), Integer(6, False), Integer(7, True), id="int6, uint6, int7"),
        pytest.param(Integer(6, False), Integer(6, True), Integer(7, True), id="uint6, int6, int7"),
        pytest.param(Integer(6, True), Integer(5, False), Integer(6, True), id="int6, uint5, int6"),
        pytest.param(Integer(5, False), Integer(6, True), Integer(6, True), id="uint5, int6, int6"),
        pytest.param(Integer(32, True), Float(32), Float(32), id="int32, float32, float32"),
        pytest.param(Integer(64, True), Float(32), Float(32), id="int64, float32, float32"),
        pytest.param(Integer(64, True), Float(64), Float(64), id="int64, float64, float64"),
        pytest.param(Integer(32, True), Float(64), Float(64), id="int32, float64, float64"),
        pytest.param(Float(64), Integer(32, True), Float(64), id="float64, int32, float64"),
        pytest.param(Float(64), Integer(7, False), Float(64), id="float64, uint7, float64"),
        pytest.param(Float(32), Float(32), Float(32), id="float32, float32, float32"),
        pytest.param(Float(32), Float(64), Float(64), id="float32, float64, float64"),
        pytest.param(Float(64), Float(32), Float(64), id="float64, float32, float64"),
        pytest.param(Float(64), Float(64), Float(64), id="float64, float64, float64"),
        pytest.param(
            UnsupportedDataType(),
            UnsupportedDataType(),
            None,
            id="unsupported, unsupported, xfail",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            Integer(6, True),
            UnsupportedDataType(),
            None,
            id="int6, unsupported, xfail",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            UnsupportedDataType(),
            Integer(6, True),
            None,
            id="unsupported, int6, xfail",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            UnsupportedDataType(),
            Float(32),
            None,
            id="unsupported, float32, xfail",
            marks=pytest.mark.xfail(strict=True),
        ),
    ],
)
def test_mix_data_types(
    dtype1: BaseDataType,
    dtype2: BaseDataType,
    expected_mixed_dtype: BaseDataType,
):
    """Test find_type_to_hold_both_lossy helper"""
    assert expected_mixed_dtype == find_type_to_hold_both_lossy(dtype1, dtype2)


@pytest.mark.parametrize(
    "value1,value2,expected_mixed_value",
    [
        pytest.param(
            EncryptedScalar(Integer(7, False)),
            EncryptedScalar(Integer(7, False)),
            EncryptedScalar(Integer(7, False)),
            id="euint7, euint7, euint7",
        ),
        pytest.param(
            EncryptedScalar(Integer(7, False)),
            ClearScalar(Integer(7, False)),
            EncryptedScalar(Integer(7, False)),
            id="euint7, cuint7, euint7",
        ),
        pytest.param(
            ClearScalar(Integer(7, False)),
            EncryptedScalar(Integer(7, False)),
            EncryptedScalar(Integer(7, False)),
            id="cuint7, euint7, euint7",
        ),
        pytest.param(
            ClearScalar(Integer(7, False)),
            ClearScalar(Integer(7, False)),
            ClearScalar(Integer(7, False)),
            id="cuint7, cuint7, cuint7",
        ),
        pytest.param(
            ClearScalar(Float(32)),
            ClearScalar(Float(32)),
            ClearScalar(Float(32)),
            id="cfloat32, cfloat32, cfloat32",
        ),
        pytest.param(
            EncryptedScalar(Float(32)),
            ClearScalar(Float(32)),
            EncryptedScalar(Float(32)),
            id="efloat32, cfloat32, efloat32",
        ),
    ],
)
def test_mix_scalar_values(value1, value2, expected_mixed_value):
    """Test mix_values_determine_holding_dtype helper with scalars"""

    assert expected_mixed_value == mix_values_determine_holding_dtype(value1, value2)


@pytest.mark.parametrize(
    "value1,value2,expected_mixed_value",
    [
        pytest.param(
            EncryptedTensor(Integer(7, False), (1, 2, 3)),
            EncryptedTensor(Integer(7, False), (1, 2, 3)),
            EncryptedTensor(Integer(7, False), (1, 2, 3)),
        ),
        pytest.param(
            ClearTensor(Integer(7, False), (1, 2, 3)),
            EncryptedTensor(Integer(7, False), (1, 2, 3)),
            EncryptedTensor(Integer(7, False), (1, 2, 3)),
        ),
        pytest.param(
            ClearTensor(Integer(7, False), (1, 2, 3)),
            ClearTensor(Integer(7, False), (1, 2, 3)),
            ClearTensor(Integer(7, False), (1, 2, 3)),
        ),
        pytest.param(
            ClearTensor(Integer(7, False), (1, 2, 3)),
            ClearTensor(Integer(7, False), (1, 2, 3)),
            ClearTensor(Integer(7, False), (1, 2, 3)),
        ),
        pytest.param(
            ClearTensor(Integer(7, False), (1, 2, 3)),
            EncryptedScalar(Integer(7, False)),
            None,
            marks=pytest.mark.xfail(raises=AssertionError),
        ),
        pytest.param(
            ClearTensor(Integer(7, False), (1, 2, 3)),
            ClearTensor(Integer(7, False), (3, 2, 1)),
            None,
            marks=pytest.mark.xfail(raises=AssertionError),
        ),
    ],
)
def test_mix_tensor_values(value1, value2, expected_mixed_value):
    """Test mix_values_determine_holding_dtype helper with tensors"""

    assert expected_mixed_value == mix_values_determine_holding_dtype(value1, value2)


class DummyValue(BaseValue):
    """DummyValue"""

    def __eq__(self, other: object) -> bool:
        return BaseValue.__eq__(self, other)


def test_fail_mix_values_determine_holding_dtype():
    """Test function for failure case of mix_values_determine_holding_dtype"""

    with pytest.raises(ValueError, match=r".* does not support value .*"):
        mix_values_determine_holding_dtype(
            DummyValue(Integer(32, True), True),
            DummyValue(Integer(32, True), True),
        )


@pytest.mark.parametrize(
    "shape1,shape2,expected_shape",
    [
        pytest.param((), (), ()),
        pytest.param((3,), (), (3,)),
        pytest.param((3,), (1,), (3,)),
        pytest.param((3,), (2,), None),
        pytest.param((3,), (3,), (3,)),
        pytest.param((2, 3), (), (2, 3)),
        pytest.param((2, 3), (1,), (2, 3)),
        pytest.param((2, 3), (2,), None),
        pytest.param((2, 3), (3,), (2, 3)),
        pytest.param((2, 3), (1, 1), (2, 3)),
        pytest.param((2, 3), (2, 1), (2, 3)),
        pytest.param((2, 3), (3, 1), None),
        pytest.param((2, 3), (1, 2), None),
        pytest.param((2, 3), (2, 2), None),
        pytest.param((2, 3), (3, 2), None),
        pytest.param((2, 3), (1, 3), (2, 3)),
        pytest.param((2, 3), (2, 3), (2, 3)),
        pytest.param((2, 3), (3, 3), None),
        pytest.param((2, 1, 3), (1, 1, 1), (2, 1, 3)),
        pytest.param((2, 1, 3), (1, 4, 1), (2, 4, 3)),
        pytest.param((2, 1, 3), (2, 4, 3), (2, 4, 3)),
        # Tests cases taken from `numpy`
        # https://github.com/numpy/numpy/blob/623bc1fae1d47df24e7f1e29321d0c0ba2771ce0/numpy/lib/tests/test_stride_tricks.py#L296-L351
        pytest.param((1, 2), (2,), (1, 2)),
        pytest.param((1, 1), (3, 4), (3, 4)),
        pytest.param((1, 3), (3, 1), (3, 3)),
        pytest.param((1, 0), (0, 0), (0, 0)),
        pytest.param((0, 1), (0, 0), (0, 0)),
        pytest.param((1, 0), (0, 1), (0, 0)),
        pytest.param((1, 1), (0, 0), (0, 0)),
        pytest.param((1, 1), (1, 0), (1, 0)),
        pytest.param((1, 1), (0, 1), (0, 1)),
        pytest.param((), (0,), (0,)),
        pytest.param((0,), (0, 0), (0, 0)),
        pytest.param((0,), (0, 1), (0, 0)),
        pytest.param((1,), (0, 0), (0, 0)),
        pytest.param((2,), (0, 0), (0, 0)),
        pytest.param((), (0, 0), (0, 0)),
        pytest.param((1, 1), (0,), (1, 0)),
        pytest.param((1,), (0, 1), (0, 1)),
        pytest.param((1,), (1, 0), (1, 0)),
        pytest.param((), (1, 0), (1, 0)),
        pytest.param((), (0, 1), (0, 1)),
        pytest.param((1,), (3,), (3,)),
        pytest.param((2,), (3, 2), (3, 2)),
        pytest.param((3,), (4,), None),
        pytest.param((2, 3), (2,), None),
        pytest.param((1, 3, 4), (2, 3, 3), None),
        pytest.param((2,), (2, 3), None),
    ],
)
def test_broadcast_shapes(shape1, shape2, expected_shape):
    """Test function for `broadcast_shapes` helper"""
    assert broadcast_shapes(shape1=shape1, shape2=shape2) == expected_shape
    assert broadcast_shapes(shape1=shape2, shape2=shape1) == expected_shape
