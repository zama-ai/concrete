"""Test file for conversion to MLIR"""
# pylint: disable=no-name-in-module,no-member
import itertools

import numpy
import pytest
from mlir.ir import IntegerType, Location, RankedTensorType, UnrankedTensorType
from zamalang import compiler
from zamalang.dialects import hlfhe

from concrete.common.data_types.integers import Integer
from concrete.common.extensions.table import LookupTable
from concrete.common.mlir import V0_OPSET_CONVERSION_FUNCTIONS, MLIRConverter
from concrete.common.values import ClearScalar, EncryptedScalar
from concrete.common.values.tensors import ClearTensor, EncryptedTensor
from concrete.numpy.compile import compile_numpy_function_into_op_graph


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
    return 12 - x


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


# TODO: remove workaround #359
def lut_more_bits_than_table_length(x, y):
    """Test lookup table when bit_width support longer LUT"""
    table = LookupTable([3, 6, 0, 2, 1, 4, 5, 7])
    return table[x] + y


# TODO: remove workaround #359
def lut_less_bits_than_table_length(x):
    """Test lookup table when bit_width support smaller LUT"""
    table = LookupTable([3, 6, 0, 2, 1, 4, 5, 7, 3, 6, 0, 2, 1, 4, 5, 7])
    return table[x]


def dot(x, y):
    """Test dot"""
    return numpy.dot(x, y)


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
            (range(0, 10),),
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
            (range(0, 10),),
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
            (range(0, 10),),
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
                "x": EncryptedScalar(Integer(3, is_signed=False)),
            },
            (range(0, 8),),
        ),
        (
            lut_more_bits_than_table_length,
            {
                "x": EncryptedScalar(Integer(64, is_signed=False)),
                "y": EncryptedScalar(Integer(64, is_signed=False)),
            },
            (range(0, 8), range(0, 16)),
        ),
        (
            lut_less_bits_than_table_length,
            {
                "x": EncryptedScalar(Integer(3, is_signed=False)),
            },
            (range(0, 8),),
        ),
    ],
)
def test_mlir_converter(func, args_dict, args_ranges):
    """Test the conversion to MLIR by calling the parser from the compiler"""
    inputset = datagen(*args_ranges)
    result_graph = compile_numpy_function_into_op_graph(func, args_dict, inputset)
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    mlir_result = converter.convert(result_graph)
    # testing that this doesn't raise an error
    compiler.round_trip(mlir_result)


@pytest.mark.parametrize(
    "func, args_dict, args_ranges",
    [
        (
            dot,
            {
                "x": EncryptedTensor(Integer(64, is_signed=False), shape=(4,)),
                "y": ClearTensor(Integer(64, is_signed=False), shape=(4,)),
            },
            (range(0, 4), range(0, 4)),
        ),
        (
            dot,
            {
                "x": ClearTensor(Integer(64, is_signed=False), shape=(4,)),
                "y": EncryptedTensor(Integer(64, is_signed=False), shape=(4,)),
            },
            (range(0, 4), range(0, 4)),
        ),
    ],
)
def test_mlir_converter_dot_between_vectors(func, args_dict, args_ranges):
    """Test the conversion to MLIR by calling the parser from the compiler"""
    assert len(args_dict["x"].shape) == 1
    assert len(args_dict["y"].shape) == 1

    n = args_dict["x"].shape[0]

    result_graph = compile_numpy_function_into_op_graph(
        func,
        args_dict,
        (([data[0]] * n, [data[1]] * n) for data in datagen(*args_ranges)),
    )
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    mlir_result = converter.convert(result_graph)
    # testing that this doesn't raise an error
    compiler.round_trip(mlir_result)


def test_concrete_encrypted_integer_to_mlir_type():
    """Test conversion of EncryptedScalar into MLIR"""
    value = EncryptedScalar(Integer(7, is_signed=False))
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    eint = converter.common_value_to_mlir_type(value)
    assert eint == hlfhe.EncryptedIntegerType.get(converter.context, 7)


@pytest.mark.parametrize("is_signed", [True, False])
def test_concrete_clear_integer_to_mlir_type(is_signed):
    """Test conversion of ClearScalar into MLIR"""
    value = ClearScalar(Integer(5, is_signed=is_signed))
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    with converter.context:
        int_mlir = converter.common_value_to_mlir_type(value)
        if is_signed:
            assert int_mlir == IntegerType.get_signed(5)
        else:
            assert int_mlir == IntegerType.get_signless(5)


@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize(
    "shape",
    [
        None,
        (5,),
        (5, 8),
        (-1, 5),
    ],
)
def test_concrete_clear_tensor_integer_to_mlir_type(is_signed, shape):
    """Test conversion of ClearTensor into MLIR"""
    value = ClearTensor(Integer(5, is_signed=is_signed), shape)
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    with converter.context, Location.unknown():
        tensor_mlir = converter.common_value_to_mlir_type(value)
        if is_signed:
            element_type = IntegerType.get_signed(5)
        else:
            element_type = IntegerType.get_signless(5)
        if shape is None:
            expected_type = UnrankedTensorType.get(element_type)
        else:
            expected_type = RankedTensorType.get(shape, element_type)
        assert tensor_mlir == expected_type


@pytest.mark.parametrize(
    "shape",
    [
        None,
        (5,),
        (5, 8),
        (-1, 5),
    ],
)
def test_concrete_encrypted_tensor_integer_to_mlir_type(shape):
    """Test conversion of EncryptedTensor into MLIR"""
    value = EncryptedTensor(Integer(6, is_signed=False), shape)
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    with converter.context, Location.unknown():
        tensor_mlir = converter.common_value_to_mlir_type(value)
        element_type = hlfhe.EncryptedIntegerType.get(converter.context, 6)
        if shape is None:
            expected_type = UnrankedTensorType.get(element_type)
        else:
            expected_type = RankedTensorType.get(shape, element_type)
        assert tensor_mlir == expected_type


def test_failing_concrete_to_mlir_type():
    """Test failing conversion of an unsupported type into MLIR"""
    value = "random"
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    with pytest.raises(TypeError, match=r"can't convert value of type .* to MLIR type"):
        converter.common_value_to_mlir_type(value)


# pylint: enable=no-name-in-module,no-member
