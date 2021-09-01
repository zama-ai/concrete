"""Test file for conversion to MLIR"""
# pylint: disable=no-name-in-module,no-member
import itertools

import pytest
from mlir.ir import IntegerType
from zamalang import compiler
from zamalang.dialects import hlfhe

from hdk.common.data_types.integers import Integer
from hdk.common.extensions.table import LookupTable
from hdk.common.mlir import V0_OPSET_CONVERSION_FUNCTIONS, MLIRConverter
from hdk.common.values import ClearScalar, EncryptedScalar
from hdk.hnumpy.compile import compile_numpy_function_into_op_graph


def add(x, y):
    """Test simple add"""
    return x + y


def constant_add(x):
    """Test constant add"""
    return x + 5


def sub(x, y):
    """Test simple sub"""
    return x - y


def constant_sub(x):
    """Test constant sub"""
    return 8 - x


def mul(x, y):
    """Test simple mul"""
    return x * y


def constant_mul(x):
    """Test constant mul"""
    return x * 2


def sub_add_mul(x, y, z):
    """Test combination of ops"""
    return z - y + x * z


def ret_multiple(x, y, z):
    """Test return of multiple values"""
    return x, y, z


def ret_multiple_different_order(x, y, z):
    """Test return of multiple values in a different order from input"""
    return y, z, x


def lut(x):
    """Test lookup table"""
    table = LookupTable([3, 6, 0, 2, 1, 4, 5, 7])
    return table[x]


def datagen(*args):
    """Generate data from ranges"""
    for prod in itertools.product(*args):
        yield prod


@pytest.mark.parametrize(
    "func, args_dict, args_ranges",
    [
        (
            add,
            {
                "x": EncryptedScalar(Integer(64, is_signed=False)),
                "y": ClearScalar(Integer(32, is_signed=False)),
            },
            (range(0, 8), range(1, 4)),
        ),
        (
            constant_add,
            {
                "x": EncryptedScalar(Integer(64, is_signed=False)),
            },
            (range(0, 8),),
        ),
        (
            add,
            {
                "x": ClearScalar(Integer(32, is_signed=False)),
                "y": EncryptedScalar(Integer(64, is_signed=False)),
            },
            (range(0, 8), range(1, 4)),
        ),
        (
            add,
            {
                "x": EncryptedScalar(Integer(7, is_signed=False)),
                "y": EncryptedScalar(Integer(7, is_signed=False)),
            },
            (range(7, 15), range(1, 5)),
        ),
        (
            sub,
            {
                "x": ClearScalar(Integer(8, is_signed=False)),
                "y": EncryptedScalar(Integer(7, is_signed=False)),
            },
            (range(5, 10), range(2, 6)),
        ),
        (
            constant_sub,
            {
                "x": EncryptedScalar(Integer(64, is_signed=False)),
            },
            (range(0, 5),),
        ),
        (
            mul,
            {
                "x": EncryptedScalar(Integer(7, is_signed=False)),
                "y": ClearScalar(Integer(8, is_signed=False)),
            },
            (range(1, 5), range(2, 8)),
        ),
        (
            constant_mul,
            {
                "x": EncryptedScalar(Integer(64, is_signed=False)),
            },
            (range(0, 8),),
        ),
        (
            mul,
            {
                "x": ClearScalar(Integer(8, is_signed=False)),
                "y": EncryptedScalar(Integer(7, is_signed=False)),
            },
            (range(1, 5), range(2, 8)),
        ),
        (
            sub_add_mul,
            {
                "x": EncryptedScalar(Integer(7, is_signed=False)),
                "y": EncryptedScalar(Integer(7, is_signed=False)),
                "z": ClearScalar(Integer(7, is_signed=False)),
            },
            (range(0, 8), range(1, 5), range(5, 12)),
        ),
        (
            ret_multiple,
            {
                "x": EncryptedScalar(Integer(7, is_signed=False)),
                "y": EncryptedScalar(Integer(7, is_signed=False)),
                "z": ClearScalar(Integer(7, is_signed=False)),
            },
            (range(1, 5), range(1, 5), range(1, 5)),
        ),
        (
            ret_multiple_different_order,
            {
                "x": EncryptedScalar(Integer(7, is_signed=False)),
                "y": EncryptedScalar(Integer(7, is_signed=False)),
                "z": ClearScalar(Integer(7, is_signed=False)),
            },
            (range(1, 5), range(1, 5), range(1, 5)),
        ),
        (
            lut,
            {
                "x": EncryptedScalar(Integer(64, is_signed=False)),
            },
            (range(0, 8),),
        ),
    ],
)
def test_mlir_converter(func, args_dict, args_ranges):
    """Test the conversion to MLIR by calling the parser from the compiler"""
    dataset = datagen(*args_ranges)
    result_graph = compile_numpy_function_into_op_graph(func, args_dict, dataset)
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    mlir_result = converter.convert(result_graph)
    # testing that this doesn't raise an error
    compiler.round_trip(mlir_result)


def test_hdk_encrypted_integer_to_mlir_type():
    """Test conversion of EncryptedScalar into MLIR"""
    value = EncryptedScalar(Integer(7, is_signed=False))
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    eint = converter.hdk_value_to_mlir_type(value)
    assert eint == hlfhe.EncryptedIntegerType.get(converter.context, 7)


@pytest.mark.parametrize("is_signed", [True, False])
def test_hdk_clear_integer_to_mlir_type(is_signed):
    """Test conversion of ClearScalar into MLIR"""
    value = ClearScalar(Integer(5, is_signed=is_signed))
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    int_mlir = converter.hdk_value_to_mlir_type(value)
    with converter.context:
        if is_signed:
            assert int_mlir == IntegerType.get_signed(5)
        else:
            assert int_mlir == IntegerType.get_signless(5)


def test_failing_hdk_to_mlir_type():
    """Test failing conversion of an unsupported type into MLIR"""
    value = "random"
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    with pytest.raises(TypeError, match=r"can't convert value of type .* to MLIR type"):
        converter.hdk_value_to_mlir_type(value)


# pylint: enable=no-name-in-module,no-member
