"""Test file for MLIR conversion helpers."""

# pylint cannot extract symbol information of 'mlir' module so we need to disable some lints

# pylint: disable=no-name-in-module

import concrete.lang as concretelang
import pytest
from mlir.ir import Context, Location

from concrete.common.data_types import Float, SignedInteger, UnsignedInteger
from concrete.common.mlir.conversion_helpers import integer_to_mlir_type, value_to_mlir_type
from concrete.common.values import ClearScalar, ClearTensor, EncryptedScalar, EncryptedTensor

# pylint: enable=no-name-in-module


@pytest.mark.parametrize(
    "integer,is_encrypted,expected_mlir_type_str",
    [
        pytest.param(SignedInteger(5), False, "i5"),
        pytest.param(UnsignedInteger(5), False, "i5"),
        pytest.param(SignedInteger(32), False, "i32"),
        pytest.param(UnsignedInteger(32), False, "i32"),
        pytest.param(SignedInteger(5), True, "!FHE.eint<5>"),
        pytest.param(UnsignedInteger(5), True, "!FHE.eint<5>"),
    ],
)
def test_integer_to_mlir_type(integer, is_encrypted, expected_mlir_type_str):
    """Test function for integer to MLIR type conversion."""

    with Context() as ctx, Location.unknown():
        concretelang.register_dialects(ctx)
        assert str(integer_to_mlir_type(ctx, integer, is_encrypted)) == expected_mlir_type_str


@pytest.mark.parametrize(
    "value,expected_mlir_type_str",
    [
        pytest.param(ClearScalar(SignedInteger(5)), "i5"),
        pytest.param(ClearTensor(SignedInteger(5), shape=(2, 3)), "tensor<2x3xi5>"),
        pytest.param(EncryptedScalar(SignedInteger(5)), "!FHE.eint<5>"),
        pytest.param(EncryptedTensor(SignedInteger(5), shape=(2, 3)), "tensor<2x3x!FHE.eint<5>>"),
        pytest.param(ClearScalar(UnsignedInteger(5)), "i5"),
        pytest.param(ClearTensor(UnsignedInteger(5), shape=(2, 3)), "tensor<2x3xi5>"),
        pytest.param(EncryptedScalar(UnsignedInteger(5)), "!FHE.eint<5>"),
        pytest.param(EncryptedTensor(UnsignedInteger(5), shape=(2, 3)), "tensor<2x3x!FHE.eint<5>>"),
    ],
)
def test_value_to_mlir_type(value, expected_mlir_type_str):
    """Test function for value to MLIR type conversion."""

    with Context() as ctx, Location.unknown():
        concretelang.register_dialects(ctx)
        assert str(value_to_mlir_type(ctx, value)) == expected_mlir_type_str


@pytest.mark.parametrize(
    "value,expected_error_message",
    [
        pytest.param(
            ClearScalar(Float(32)),
            "ClearScalar<float32> is not supported for MLIR conversion",
        ),
        pytest.param(
            ClearTensor(Float(32), shape=(2, 3)),
            "ClearTensor<float32, shape=(2, 3)> is not supported for MLIR conversion",
        ),
        pytest.param(
            EncryptedScalar(Float(32)),
            "EncryptedScalar<float32> is not supported for MLIR conversion",
        ),
        pytest.param(
            EncryptedTensor(Float(32), shape=(2, 3)),
            "EncryptedTensor<float32, shape=(2, 3)> is not supported for MLIR conversion",
        ),
    ],
)
def test_fail_value_to_mlir_type(value, expected_error_message):
    """Test function for failed value to MLIR type conversion."""

    with pytest.raises(TypeError) as excinfo:
        with Context() as ctx, Location.unknown():
            concretelang.register_dialects(ctx)
            value_to_mlir_type(ctx, value)

    assert str(excinfo.value) == expected_error_message
