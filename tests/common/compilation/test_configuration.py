"""Test file for compilation configuration"""

from inspect import signature

import numpy
import pytest

from concrete.common.compilation import CompilationConfiguration
from concrete.common.data_types.integers import Integer
from concrete.common.values import EncryptedScalar
from concrete.numpy.compile import compile_numpy_function_into_op_graph


def no_fuse(x):
    """No fuse"""
    return x + 2


def simple_fuse_not_output(x):
    """Simple fuse not output"""
    intermediate = x.astype(numpy.float64)
    intermediate = intermediate.astype(numpy.uint32)
    return intermediate + 2


@pytest.mark.parametrize(
    "function_to_trace,fused",
    [
        pytest.param(
            no_fuse,
            False,
            id="no_fuse",
        ),
        pytest.param(
            simple_fuse_not_output,
            True,
            id="simple_fuse_not_output",
            marks=pytest.mark.xfail(strict=True),
            # fails because it connot be compiled without topological optimizations
        ),
    ],
)
def test_enable_topological_optimizations(
    test_helpers, function_to_trace, fused, default_compilation_configuration
):
    """Test function for enable_topological_optimizations flag of compilation configuration"""

    op_graph = compile_numpy_function_into_op_graph(
        function_to_trace,
        {
            param: EncryptedScalar(Integer(32, is_signed=False))
            for param in signature(function_to_trace).parameters.keys()
        },
        [(i,) for i in range(10)],
        default_compilation_configuration,
    )
    op_graph_not_optimized = compile_numpy_function_into_op_graph(
        function_to_trace,
        {
            param: EncryptedScalar(Integer(32, is_signed=False))
            for param in signature(function_to_trace).parameters.keys()
        },
        [(i,) for i in range(10)],
        CompilationConfiguration(
            dump_artifacts_on_unexpected_failures=False,
            enable_topological_optimizations=False,
            treat_warnings_as_errors=True,
        ),
    )

    graph = op_graph.graph
    not_optimized_graph = op_graph_not_optimized.graph

    if fused:
        assert not test_helpers.digraphs_are_equivalent(graph, not_optimized_graph)
        assert len(graph) < len(not_optimized_graph)
    else:
        assert test_helpers.digraphs_are_equivalent(graph, not_optimized_graph)
        assert len(graph) == len(not_optimized_graph)
