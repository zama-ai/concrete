"""
Tests of `Graph` class.
"""

import os
import re

import numpy as np
import pytest

import tests
from concrete import fhe

tests_directory = os.path.dirname(tests.__file__)


def g(z):
    """
    Example function with a tag.
    """

    with fhe.tag("def"):
        a = 120 - z
        b = a // 4
    return b


def f(x):
    """
    Example function with nested tags.
    """

    with fhe.tag("abc"):
        x = x * 2
        with fhe.tag("foo"):
            y = x + 42
        z = np.sqrt(y).astype(np.int64)

    return g(z + 3) * 2


def test_graph_format_show_lines(helpers):
    """
    Test `format` method of `Graph` class with show_lines=True.
    """

    configuration = helpers.configuration()

    compiler = fhe.Compiler(f, {"x": "encrypted"})
    graph = compiler.trace(range(10), configuration)

    # pylint: disable=line-too-long
    expected = f"""

 %0 = x                            # EncryptedScalar<uint4>        ∈ [0, 9]                             {tests_directory}/representation/test_graph.py:50
 %1 = 2                            # ClearScalar<uint2>            ∈ [2, 2]            @ abc            {tests_directory}/representation/test_graph.py:34
 %2 = multiply(%0, %1)             # EncryptedScalar<uint5>        ∈ [0, 18]           @ abc            {tests_directory}/representation/test_graph.py:34
 %3 = 42                           # ClearScalar<uint6>            ∈ [42, 42]          @ abc.foo        {tests_directory}/representation/test_graph.py:36
 %4 = add(%2, %3)                  # EncryptedScalar<uint6>        ∈ [42, 60]          @ abc.foo        {tests_directory}/representation/test_graph.py:36
 %5 = subgraph(%4)                 # EncryptedScalar<uint3>        ∈ [6, 7]            @ abc            {tests_directory}/representation/test_graph.py:37
 %6 = 3                            # ClearScalar<uint2>            ∈ [3, 3]                             {tests_directory}/representation/test_graph.py:39
 %7 = add(%5, %6)                  # EncryptedScalar<uint4>        ∈ [9, 10]                            {tests_directory}/representation/test_graph.py:39
 %8 = 120                          # ClearScalar<uint7>            ∈ [120, 120]        @ def            {tests_directory}/representation/test_graph.py:23
 %9 = subtract(%8, %7)             # EncryptedScalar<uint7>        ∈ [110, 111]        @ def            {tests_directory}/representation/test_graph.py:23
%10 = 4                            # ClearScalar<uint3>            ∈ [4, 4]            @ def            {tests_directory}/representation/test_graph.py:24
%11 = floor_divide(%9, %10)        # EncryptedScalar<uint5>        ∈ [27, 27]          @ def            {tests_directory}/representation/test_graph.py:24
%12 = 2                            # ClearScalar<uint2>            ∈ [2, 2]                             {tests_directory}/representation/test_graph.py:39
%13 = multiply(%11, %12)           # EncryptedScalar<uint6>        ∈ [54, 54]                           {tests_directory}/representation/test_graph.py:39
return %13

Subgraphs:

    %5 = subgraph(%4):

        %0 = input                         # EncryptedScalar<uint2>          @ abc.foo        {tests_directory}/representation/test_graph.py:36
        %1 = sqrt(%0)                      # EncryptedScalar<float64>        @ abc            {tests_directory}/representation/test_graph.py:37
        %2 = astype(%1, dtype=int_)        # EncryptedScalar<uint1>          @ abc            {tests_directory}/representation/test_graph.py:37
        return %2

    """  # noqa: E501
    # pylint: enable=line-too-long

    actual = graph.format(show_locations=True)

    assert (
        actual.strip() == expected.strip()
    ), f"""

Expected Output
===============
{expected}

Actual Output
=============
{actual}

            """


@pytest.mark.parametrize(
    "function,inputset,kwargs,expected_result",
    [
        pytest.param(
            lambda x: x + 1,
            range(5),
            {},
            3,
        ),
        pytest.param(
            lambda x: x + 42,
            range(10),
            {},
            6,
        ),
        pytest.param(
            lambda x: x + 42,
            range(50),
            {},
            7,
        ),
        pytest.param(
            f,
            range(10),
            {},
            7,
        ),
        pytest.param(
            f,
            range(10),
            {"tag_filter": ""},
            6,
        ),
        pytest.param(
            f,
            range(10),
            {"tag_filter": "abc"},
            5,
        ),
        pytest.param(
            f,
            range(10),
            {"tag_filter": ["abc", "def"]},
            7,
        ),
        pytest.param(
            f,
            range(10),
            {"tag_filter": re.compile(".*b.*")},
            6,
        ),
        pytest.param(
            f,
            range(10),
            {"operation_filter": "input"},
            4,
        ),
        pytest.param(
            f,
            range(10),
            {"operation_filter": "constant"},
            7,
        ),
        pytest.param(
            f,
            range(10),
            {"operation_filter": "subgraph"},
            3,
        ),
        pytest.param(
            f,
            range(10),
            {"operation_filter": "add"},
            6,
        ),
        pytest.param(
            f,
            range(10),
            {"operation_filter": ["subgraph", "add"]},
            6,
        ),
        pytest.param(
            f,
            range(10),
            {"operation_filter": re.compile("sub.*")},
            7,
        ),
        pytest.param(
            f,
            range(10),
            {"tag_filter": "abc.foo", "operation_filter": "add"},
            6,
        ),
        pytest.param(
            f,
            range(10),
            {"tag_filter": "abc", "operation_filter": "floor_divide"},
            -1,
        ),
        pytest.param(
            lambda x: (x**2) * 30,
            range(10),
            {"operation_filter": "power"},
            7,
        ),
        pytest.param(
            lambda x: (x**2) * 30,
            range(10),
            {"operation_filter": "power", "assigned_bit_width": True},
            12,
        ),
    ],
)
def test_graph_maximum_integer_bit_width(
    function,
    inputset,
    kwargs,
    expected_result,
    helpers,
):
    """
    Test `maximum_integer_bit_width` method of `Graph` class.
    """

    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, {"x": "encrypted"})
    circuit = compiler.compile(inputset, configuration)

    assert circuit.graph.maximum_integer_bit_width(**kwargs) == expected_result


@pytest.mark.parametrize(
    "function,inputset,tag_filter,operation_filter,is_encrypted_filter,expected_result",
    [
        pytest.param(
            lambda x: x + 42,
            range(-10, 10),
            None,
            None,
            None,
            (-10, 51),
        ),
        pytest.param(
            lambda x: x + 1.2,
            [1.5, 4.2],
            None,
            None,
            None,
            None,
        ),
        pytest.param(
            f,
            range(10),
            None,
            None,
            None,
            (0, 120),
        ),
        pytest.param(
            f,
            range(10),
            "",
            None,
            None,
            (0, 54),
        ),
        pytest.param(
            f,
            range(10),
            "abc",
            None,
            None,
            (0, 18),
        ),
        pytest.param(
            f,
            range(10),
            ["abc", "def"],
            None,
            None,
            (0, 120),
        ),
        pytest.param(
            f,
            range(10),
            re.compile(".*b.*"),
            None,
            None,
            (0, 60),
        ),
        pytest.param(
            f,
            range(10),
            None,
            "input",
            None,
            (0, 9),
        ),
        pytest.param(
            f,
            range(10),
            None,
            "constant",
            None,
            (2, 120),
        ),
        pytest.param(
            f,
            range(10),
            None,
            "subgraph",
            None,
            (6, 7),
        ),
        pytest.param(
            f,
            range(10),
            None,
            "add",
            None,
            (9, 60),
        ),
        pytest.param(
            f,
            range(10),
            None,
            ["subgraph", "add"],
            None,
            (6, 60),
        ),
        pytest.param(
            f,
            range(10),
            None,
            re.compile("sub.*"),
            None,
            (6, 111),
        ),
        pytest.param(
            f,
            range(10),
            "abc.foo",
            "add",
            None,
            (42, 60),
        ),
        pytest.param(
            f,
            range(10),
            "abc",
            "floor_divide",
            None,
            None,
        ),
        pytest.param(
            lambda x: x - 2,
            range(5, 10),
            None,
            None,
            True,
            (3, 9),
        ),
        pytest.param(
            lambda x: x - 2,
            range(5, 10),
            None,
            None,
            False,
            (2, 2),
        ),
    ],
)
def test_graph_integer_range(
    function,
    inputset,
    tag_filter,
    operation_filter,
    is_encrypted_filter,
    expected_result,
    helpers,
):
    """
    Test `integer_range` method of `Graph` class.
    """

    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, {"x": "encrypted"})
    graph = compiler.trace(inputset, configuration)

    assert graph.integer_range(tag_filter, operation_filter, is_encrypted_filter) == expected_result


def test_direct_graph_integer_range(helpers):
    """
    Test `integer_range` method of `Graph` class where `graph.is_direct` is `True`.
    """

    # pylint: disable=import-outside-toplevel
    from concrete.fhe.dtypes import Integer
    from concrete.fhe.values import ValueDescription

    # pylint: enable=import-outside-toplevel

    circuit = fhe.Compiler.assemble(
        lambda x: x,
        {
            "x": ValueDescription(
                dtype=Integer(is_signed=False, bit_width=8), shape=(), is_encrypted=True
            )
        },
        configuration=helpers.configuration(),
    )
    assert circuit.graph.integer_range() is None


def test_graph_processors(helpers):
    """
    Test providing additional graph processors.
    """

    class RecordInputBitWidth(fhe.GraphProcessor):
        """Sample graph processor to record the input bit width."""

        input_bit_width: int = 0

        def apply(self, graph: fhe.Graph):
            assert len(graph.input_nodes) == 1
            assert isinstance(graph.input_nodes[0].output.dtype, fhe.Integer)
            self.input_bit_width = graph.input_nodes[0].output.dtype.bit_width

    class CountNodes(fhe.GraphProcessor):
        """Sample graph processor to count nodes."""

        node_count: int = 0

        def apply(self, graph: fhe.Graph):
            self.node_count += len(graph.query_nodes())

    pre_processor1 = RecordInputBitWidth()
    post_processor1 = RecordInputBitWidth()
    post_processor2 = CountNodes()
    configuration = helpers.configuration().fork(
        additional_pre_processors=[pre_processor1],
        additional_post_processors=[post_processor1, post_processor2],
    )

    compiler = fhe.Compiler(lambda x: (x + 5) ** 2, {"x": "encrypted"})
    compiler.compile(range(8), configuration)

    assert pre_processor1.input_bit_width == 3
    assert post_processor1.input_bit_width == 8 if configuration.single_precision else 4
    assert post_processor2.node_count == 5


@pytest.mark.parametrize(
    "function,encryption_status,inputset,expected_inputs_count,expected_outputs_count",
    [
        pytest.param(
            lambda x: (x + 1, x * 2),
            {"x": "encrypted"},
            fhe.inputset(fhe.uint4),
            1,
            2,
        ),
        pytest.param(
            lambda x, y: x + y,
            {"x": "encrypted", "y": "encrypted"},
            fhe.inputset(fhe.uint4, fhe.uint4),
            2,
            1,
        ),
        pytest.param(
            lambda x, y, z: (x + y, x + z, y + z),
            {"x": "encrypted", "y": "encrypted", "z": "encrypted"},
            fhe.inputset(fhe.uint4, fhe.uint4, fhe.uint4),
            3,
            3,
        ),
    ],
)
def test_graph_inputs_outputs_count(
    function,
    encryption_status,
    inputset,
    expected_inputs_count,
    expected_outputs_count,
    helpers,
):
    """
    Test `inputs_count` property of `Graph`.
    """

    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, encryption_status)
    graph = compiler.trace(inputset, configuration)

    assert graph.inputs_count == expected_inputs_count
    assert graph.outputs_count == expected_outputs_count
