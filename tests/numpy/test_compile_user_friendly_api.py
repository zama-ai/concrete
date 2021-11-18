"""Test file for user-friendly numpy compilation functions"""

import numpy
import pytest

from concrete.common.debugging import format_operation_graph
from concrete.numpy.np_fhe_compiler import NPFHECompiler


def complicated_topology(x, y):
    """Mix x in an intricated way."""
    intermediate = x + y
    x_p_1 = intermediate + 1
    x_p_2 = intermediate + 2
    x_p_3 = x_p_1 + x_p_2
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
    )


@pytest.mark.parametrize("input_shape", [(), (3, 1, 2)])
def test_np_fhe_compiler(input_shape, default_compilation_configuration):
    """Test NPFHECompiler in two subtests."""
    subtest_np_fhe_compiler_1_input(input_shape, default_compilation_configuration)
    subtest_np_fhe_compiler_2_inputs(input_shape, default_compilation_configuration)


def subtest_np_fhe_compiler_1_input(input_shape, default_compilation_configuration):
    """test for NPFHECompiler on one input function"""

    compiler = NPFHECompiler(
        lambda x: complicated_topology(x, 0),
        {"x": "encrypted"},
        default_compilation_configuration,
    )

    # For coverage when the OPGraph is not yet traced
    compiler._patch_op_graph_input_to_accept_any_integer_input()  # pylint: disable=protected-access

    assert compiler.compilation_configuration == default_compilation_configuration
    assert compiler.compilation_configuration is not default_compilation_configuration

    for i in numpy.arange(5):
        i = numpy.ones(input_shape, dtype=numpy.int64) * i
        assert numpy.array_equal(compiler(i), complicated_topology(i, 0))

    # For coverage, check that we flush the inputset when we query the OPGraph
    current_op_graph = compiler.op_graph
    assert current_op_graph is not compiler.op_graph
    assert len(compiler._current_inputset) == 0  # pylint: disable=protected-access
    # For coverage, cover case where the current inputset is empty
    compiler._eval_on_current_inputset()  # pylint: disable=protected-access

    # Continue a bit more
    for i in numpy.arange(5, 10):
        i = numpy.ones(input_shape, dtype=numpy.int64) * i
        assert numpy.array_equal(compiler(i), complicated_topology(i, 0))

    if input_shape == ():
        assert (
            (got := format_operation_graph(compiler.op_graph))
            == """ %0 = 67                              # ClearScalar<uint7>
 %1 = 2                               # ClearScalar<uint2>
 %2 = 3                               # ClearScalar<uint2>
 %3 = 1                               # ClearScalar<uint1>
 %4 = x                               # EncryptedScalar<uint4>
 %5 = 0                               # ClearScalar<uint1>
 %6 = add(%4, %5)                     # EncryptedScalar<uint4>
 %7 = add(%6, %1)                     # EncryptedScalar<uint4>
 %8 = add(%6, %3)                     # EncryptedScalar<uint4>
 %9 = astype(%7, dtype=int32)         # EncryptedScalar<uint4>
%10 = add(%7, %2)                     # EncryptedScalar<uint4>
%11 = add(%8, %7)                     # EncryptedScalar<uint5>
%12 = astype(%10, dtype=int32)        # EncryptedScalar<uint4>
%13 = astype(%11, dtype=int32)        # EncryptedScalar<uint5>
%14 = astype(%11, dtype=int32)        # EncryptedScalar<uint5>
%15 = add(%14, %0)                    # EncryptedScalar<uint7>
(%13, %9, %12, %15)"""
        ), got
    else:
        assert (
            (got := format_operation_graph(compiler.op_graph))
            == """ %0 = 67                              # ClearScalar<uint7>
 %1 = 2                               # ClearScalar<uint2>
 %2 = 3                               # ClearScalar<uint2>
 %3 = 1                               # ClearScalar<uint1>
 %4 = x                               # EncryptedTensor<uint4, shape=(3, 1, 2)>
 %5 = 0                               # ClearScalar<uint1>
 %6 = add(%4, %5)                     # EncryptedTensor<uint4, shape=(3, 1, 2)>
 %7 = add(%6, %1)                     # EncryptedTensor<uint4, shape=(3, 1, 2)>
 %8 = add(%6, %3)                     # EncryptedTensor<uint4, shape=(3, 1, 2)>
 %9 = astype(%7, dtype=int32)         # EncryptedTensor<uint4, shape=(3, 1, 2)>
%10 = add(%7, %2)                     # EncryptedTensor<uint4, shape=(3, 1, 2)>
%11 = add(%8, %7)                     # EncryptedTensor<uint5, shape=(3, 1, 2)>
%12 = astype(%10, dtype=int32)        # EncryptedTensor<uint4, shape=(3, 1, 2)>
%13 = astype(%11, dtype=int32)        # EncryptedTensor<uint5, shape=(3, 1, 2)>
%14 = astype(%11, dtype=int32)        # EncryptedTensor<uint5, shape=(3, 1, 2)>
%15 = add(%14, %0)                    # EncryptedTensor<uint7, shape=(3, 1, 2)>
(%13, %9, %12, %15)"""
        ), got


def subtest_np_fhe_compiler_2_inputs(input_shape, default_compilation_configuration):
    """test for NPFHECompiler on two inputs function"""

    compiler = NPFHECompiler(
        complicated_topology,
        {"x": "encrypted", "y": "clear"},
        default_compilation_configuration,
    )

    # For coverage when the OPGraph is not yet traced
    compiler._patch_op_graph_input_to_accept_any_integer_input()  # pylint: disable=protected-access

    assert compiler.compilation_configuration == default_compilation_configuration
    assert compiler.compilation_configuration is not default_compilation_configuration

    for i, j in zip(numpy.arange(5), numpy.arange(5, 10)):
        i = numpy.ones(input_shape, dtype=numpy.int64) * i
        j = numpy.ones(input_shape, dtype=numpy.int64) * j
        assert numpy.array_equal(compiler(i, j), complicated_topology(i, j))

    # For coverage, check that we flush the inputset when we query the OPGraph
    current_op_graph = compiler.op_graph
    assert current_op_graph is not compiler.op_graph
    assert len(compiler._current_inputset) == 0  # pylint: disable=protected-access
    # For coverage, cover case where the current inputset is empty
    compiler._eval_on_current_inputset()  # pylint: disable=protected-access

    # Continue a bit more
    for i, j in zip(numpy.arange(5, 10), numpy.arange(5)):
        i = numpy.ones(input_shape, dtype=numpy.int64) * i
        j = numpy.ones(input_shape, dtype=numpy.int64) * j
        assert numpy.array_equal(compiler(i, j), complicated_topology(i, j))

    if input_shape == ():
        assert (
            (got := format_operation_graph(compiler.op_graph))
            == """ %0 = 67                              # ClearScalar<uint7>
 %1 = 2                               # ClearScalar<uint2>
 %2 = 3                               # ClearScalar<uint2>
 %3 = 1                               # ClearScalar<uint1>
 %4 = x                               # EncryptedScalar<uint4>
 %5 = y                               # ClearScalar<uint4>
 %6 = add(%4, %5)                     # EncryptedScalar<uint4>
 %7 = add(%6, %1)                     # EncryptedScalar<uint4>
 %8 = add(%6, %3)                     # EncryptedScalar<uint4>
 %9 = astype(%7, dtype=int32)         # EncryptedScalar<uint4>
%10 = add(%7, %2)                     # EncryptedScalar<uint5>
%11 = add(%8, %7)                     # EncryptedScalar<uint5>
%12 = astype(%10, dtype=int32)        # EncryptedScalar<uint5>
%13 = astype(%11, dtype=int32)        # EncryptedScalar<uint5>
%14 = astype(%11, dtype=int32)        # EncryptedScalar<uint5>
%15 = add(%14, %0)                    # EncryptedScalar<uint7>
(%13, %9, %12, %15)"""
        ), got
    else:
        assert (
            (got := format_operation_graph(compiler.op_graph))
            == """ %0 = 67                              # ClearScalar<uint7>
 %1 = 2                               # ClearScalar<uint2>
 %2 = 3                               # ClearScalar<uint2>
 %3 = 1                               # ClearScalar<uint1>
 %4 = x                               # EncryptedTensor<uint4, shape=(3, 1, 2)>
 %5 = y                               # ClearTensor<uint4, shape=(3, 1, 2)>
 %6 = add(%4, %5)                     # EncryptedTensor<uint4, shape=(3, 1, 2)>
 %7 = add(%6, %1)                     # EncryptedTensor<uint4, shape=(3, 1, 2)>
 %8 = add(%6, %3)                     # EncryptedTensor<uint4, shape=(3, 1, 2)>
 %9 = astype(%7, dtype=int32)         # EncryptedTensor<uint4, shape=(3, 1, 2)>
%10 = add(%7, %2)                     # EncryptedTensor<uint5, shape=(3, 1, 2)>
%11 = add(%8, %7)                     # EncryptedTensor<uint5, shape=(3, 1, 2)>
%12 = astype(%10, dtype=int32)        # EncryptedTensor<uint5, shape=(3, 1, 2)>
%13 = astype(%11, dtype=int32)        # EncryptedTensor<uint5, shape=(3, 1, 2)>
%14 = astype(%11, dtype=int32)        # EncryptedTensor<uint5, shape=(3, 1, 2)>
%15 = add(%14, %0)                    # EncryptedTensor<uint7, shape=(3, 1, 2)>
(%13, %9, %12, %15)"""
        ), got


def remaining_inputset_size(inputset_len):
    """Small function to generate test cases below for remaining inputset length."""
    return inputset_len % NPFHECompiler.INPUTSET_SIZE_BEFORE_AUTO_BOUND_UPDATE


@pytest.mark.parametrize(
    "inputset_len, expected_remaining_inputset_len",
    [
        (42, remaining_inputset_size(42)),
        (128, remaining_inputset_size(128)),
        (234, remaining_inputset_size(234)),
    ],
)
def test_np_fhe_compiler_auto_flush(
    inputset_len,
    expected_remaining_inputset_len,
    default_compilation_configuration,
):
    """Test the auto flush of NPFHECompiler once the inputset is 128 elements."""
    compiler = NPFHECompiler(
        lambda x: x // 2,
        {"x": "encrypted"},
        default_compilation_configuration,
    )

    for i in numpy.arange(inputset_len):
        assert numpy.array_equal(compiler(i), i // 2)

    # Check the inputset was properly flushed
    assert (
        len(compiler._current_inputset)  # pylint: disable=protected-access
        == expected_remaining_inputset_len
    )
