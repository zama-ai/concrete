"""Benchmark module for the entire compilation pipeline"""

import itertools

import pytest

import concrete.numpy as hnp


@pytest.mark.parametrize(
    "function,parameters,ranges",
    [
        pytest.param(
            lambda x: x + 42,
            {"x": hnp.EncryptedScalar(hnp.SignedInteger(4))},
            ((-2, 2),),
            id="x + 42",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": hnp.EncryptedScalar(hnp.SignedInteger(4)),
                "y": hnp.EncryptedScalar(hnp.UnsignedInteger(4)),
            },
            ((-2, 2), (20, 30)),
            id="x + y",
        ),
    ],
)
def test_compilation(benchmark, function, parameters, ranges):
    """Benchmark function for compilation of various functions"""

    def dataset(args):
        for prod in itertools.product(*args):
            yield prod

    @benchmark
    def compilation():
        hnp.compile_numpy_function_into_op_graph(function, parameters, dataset(ranges))


@pytest.mark.parametrize(
    "function,parameters,ranges,inputs",
    [
        pytest.param(
            lambda x: x + 420,
            {"x": hnp.EncryptedScalar(hnp.SignedInteger(4))},
            ((-2, 2),),
            [
                {0: -2},
                {0: 0},
                {0: 1},
            ],
            id="x + 420",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": hnp.EncryptedScalar(hnp.SignedInteger(4)),
                "y": hnp.EncryptedScalar(hnp.UnsignedInteger(4)),
            },
            ((-2, 2), (20, 30)),
            [
                {0: -2, 1: 25},
                {0: 0, 1: 30},
                {0: 1, 1: 22},
            ],
            id="x + y",
        ),
    ],
)
def test_evaluation(benchmark, function, parameters, ranges, inputs):
    """Benchmark function for evaluation of various functions"""

    def dataset(args):
        for prod in itertools.product(*args):
            yield prod

    graph = hnp.compile_numpy_function_into_op_graph(function, parameters, dataset(ranges))

    @benchmark
    def evaluation():
        for x in inputs:
            graph.evaluate(x)
