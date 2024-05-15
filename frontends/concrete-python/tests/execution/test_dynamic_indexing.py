"""
Tests of execution of dynamic indexing operation.
"""

import random

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "encryption_status,function,inputset",
    [
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3], lambda _: np.random.randint(0, 3)),
            id="x[y] where x.shape == (3,) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3], lambda _: np.random.randint(-3, 3)),
            id="x[y] where x.shape == (3,) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 31], lambda _: np.random.randint(0, 3)),
            id="x[y] where x.shape == (31,) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 31], lambda _: np.random.randint(-3, 3)),
            id="x[y] where x.shape == (31,) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 0],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 3)),
            id="x[y, 0] where x.shape == (3, 4) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 0],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-3, 3)),
            id="x[y, 0] where x.shape == (3, 4) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 4)),
            id="x[1, y] where x.shape == (3, 4) | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-4, 4)),
            id="x[1, y] where x.shape == (3, 4) | -4 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4],
                lambda _: np.random.randint(0, 3),
                lambda _: np.random.randint(0, 4),
            ),
            id="x[y, z] where x.shape == (3, 4) | 0 < y < 3 | 0 < z < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4],
                lambda _: np.random.randint(0, 3),
                lambda _: np.random.randint(-4, 4),
            ),
            id="x[y, z] where x.shape == (3, 4) | 0 < y < 3 | -4 < z < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4],
                lambda _: np.random.randint(-3, 3),
                lambda _: np.random.randint(0, 4),
            ),
            id="x[y, z] where x.shape == (3, 4) | -3 < y < 3 | 0 < z < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4],
                lambda _: np.random.randint(-3, 3),
                lambda _: np.random.randint(-4, 4),
            ),
            id="x[y, z] where x.shape == (3, 4) | -3 < y < 3 | -4 < z < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 3)),
            id="x[y] where x.shape == (3, 4) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-3, 3)),
            id="x[y] where x.shape == (3, 4) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[:, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 4)),
            id="x[:, y] where x.shape == (3, 4) | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[:, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-4, 4)),
            id="x[:, y] where x.shape == (3, 4) | -4 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 2],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 3)),
            id="x[y, 2] where x.shape == (3, 4) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 2],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-3, 3)),
            id="x[y, 2] where x.shape == (3, 4) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, -2],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 3)),
            id="x[y, -2] where x.shape == (3, 4) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, -2],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-3, 3)),
            id="x[y, -2] where x.shape == (3, 4) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 4)),
            id="x[1, y] where x.shape == (3, 4) | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-4, 4)),
            id="x[1, y] where x.shape == (3, 4) | -4 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[-1, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 4)),
            id="x[-1, y] where x.shape == (3, 4) | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[-1, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-4, 4)),
            id="x[-1, y] where x.shape == (3, 4) | -4 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, 4:6],
            fhe.inputset(fhe.tensor[fhe.uint4, 10, 9, 8], lambda _: np.random.randint(0, 9)),
            id="x[1:3, y, 4:6] where x.shape == (10, 9, 8) | 0 < y < 9",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, 4:6],
            fhe.inputset(fhe.tensor[fhe.uint4, 10, 9, 8], lambda _: np.random.randint(-9, 9)),
            id="x[1:3, y, 4:6] where x.shape == (10, 9, 8) | -9 < y < 9",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, 0, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 10, 9, 8],
                lambda _: np.random.randint(0, 10),
                lambda _: np.random.randint(0, 8),
            ),
            id="x[y, 0, z] where x.shape == (10, 9, 8) | 0 < y < 10 | 0 < z < 8",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, 0, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 10, 9, 8],
                lambda _: np.random.randint(0, 10),
                lambda _: np.random.randint(-8, 8),
            ),
            id="x[y, 0, z] where x.shape == (10, 9, 8) | 0 < y < 10 | -8 < z < 8",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, 0, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 10, 9, 8],
                lambda _: np.random.randint(-10, 10),
                lambda _: np.random.randint(0, 8),
            ),
            id="x[y, 0, z] where x.shape == (10, 9, 8) | -10 < y < 10 | 0 < z < 8",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, 0, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 10, 9, 8],
                lambda _: np.random.randint(-10, 10),
                lambda _: np.random.randint(-8, 8),
            ),
            id="x[y, 0, z] where x.shape == (10, 9, 8) | -10 < y < 10 | -8 < z < 8",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 5], lambda _: np.random.randint(0, 5, size=(3,))),
            id="x[y] where x.shape == (5,) | 0 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 5], lambda _: np.random.randint(-5, 5, size=(3,))),
            id="x[y] where x.shape == (5,) | -5 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 5], lambda _: np.random.randint(0, 5, size=(3, 2))),
            id="x[y] where x.shape == (5,) | 0 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 5], lambda _: np.random.randint(-5, 5, size=(3, 2))),
            id="x[y] where x.shape == (5,) | -5 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=(3,)),
            ),
            id="x[y, 1] where x.shape == (5, 3) | 0 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(-5, 5, size=(3,)),
            ),
            id="x[y, 1] where x.shape == (5, 3) | -5 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=(3, 2)),
            ),
            id="x[y, 1] where x.shape == (5, 3) | -5 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=(3, 2)),
            ),
            id="x[y, 1] where x.shape == (5, 3) | -5 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, [2, 0]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=()),
            ),
            id="x[y, [2, 0]] where x.shape == (5, 3) | 0 < y < 5",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, [2, 0]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(-5, 5, size=()),
            ),
            id="x[y, [2, 0]] where x.shape == (5, 3) | -5 < y < 5",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, [2, 0]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=(2,)),
            ),
            id="x[y, [2, 0]] where x.shape == (5, 3) | 0 < y < 5 where y.shape == (2,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, [2, 0]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(-5, 5, size=(2,)),
            ),
            id="x[y, [2, 0]] where x.shape == (5, 3) | -5 < y < 5 where y.shape == (2,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=(2,)),
                lambda _: np.random.randint(0, 3, size=(2,)),
            ),
            id=(
                "x[y, z] where x.shape == (5, 3) "
                "| 0 < y < 5 where y.shape == (2,) "
                "| 0 < z < 3 where z.shape == (2,)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=(2,)),
                lambda _: np.random.randint(-3, 3, size=(2,)),
            ),
            id=(
                "x[y, z] where x.shape == (5, 3)"
                "| 0 < y < 5 where y.shape == (2,) "
                "| -3 < z < 3 where z.shape == (2,)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(-5, 5, size=(2,)),
                lambda _: np.random.randint(0, 3, size=(2,)),
            ),
            id=(
                "x[y, z] where x.shape == (5, 3)"
                "| -5 < y < 5 where y.shape == (2,) "
                "| 0 < z < 3 where z.shape == (2,)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(-5, 5, size=(2,)),
                lambda _: np.random.randint(-3, 3, size=(2,)),
            ),
            id=(
                "x[y, z] where x.shape == (5, 3) "
                "| -5 < y < 5 where y.shape == (2,) "
                "| -3 < z < 3 where z.shape == (2,)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=(3, 2)),
                lambda _: np.random.randint(0, 3, size=(3, 2)),
            ),
            id=(
                "x[y, z] where x.shape == (5, 3) | 0 < y < 5 where y.shape == (3, 2) "
                "| 0 < z < 3 where z.shape == (3, 2)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(0, 5, size=(3, 2)),
                lambda _: np.random.randint(-3, 3, size=(3, 2)),
            ),
            id=(
                "x[y, z] where x.shape == (5, 3) "
                "| 0 < y < 5 where y.shape == (3, 2) "
                "| -3 < z < 3 where z.shape == (3, 2)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(-5, 5, size=(3, 2)),
                lambda _: np.random.randint(0, 3, size=(3, 2)),
            ),
            id=(
                "x[y, z] where x.shape == (5, 3) "
                "| -5 < y < 5 where y.shape == (3, 2) "
                "| 0 < z < 3 where z.shape == (3, 2)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 3],
                lambda _: np.random.randint(-5, 5, size=(3, 2)),
                lambda _: np.random.randint(-3, 3, size=(3, 2)),
            ),
            id=(
                "x[y, z] where x.shape == (5, 3) "
                "| -5 < y < 5 where y.shape == (3, 2) "
                "| -3 < z < 3 where z.shape == (3, 2)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 6],
                lambda _: np.random.randint(0, 5, size=(3,)),
            ),
            id="x[y, 2:4] where x.shape == (5, 6) | 0 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 5, 6],
                lambda _: np.random.randint(-5, 5, size=(3,)),
            ),
            id="x[y, 2:4] where x.shape == (5, 6) | -5 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[3, y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 5, size=(3,)),
            ),
            id="x[3, y, 2:4] where x.shape == (4, 5, 6) | 0 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[3, y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-5, 5, size=(3,)),
            ),
            id="x[3, y, 2:4] where x.shape == (4, 5, 6) | -5 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, 4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 5, size=(3,)),
            ),
            id="x[1:3, y, 4] where x.shape == (4, 5, 6) | 0 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, 4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-5, 5, size=(3,)),
            ),
            id="x[1:3, y, 4] where x.shape == (4, 5, 6) | -5 < y < 5 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 3, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 4, size=(3,)),
            ),
            id="x[y, 3, 2:4] where x.shape == (4, 5, 6) | 0 < y < 4 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 3, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-4, 4, size=(3,)),
            ),
            id="x[y, 3, 2:4] where x.shape == (4, 5, 6) | -4 < y < 4 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1:3, 4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 4, size=(3,)),
            ),
            id="x[y, 1:3, 4] where x.shape == (4, 5, 6) | 0 < y < 4 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1:3, 4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-4, 4, size=(3,)),
            ),
            id="x[y, 1:3, 4] where x.shape == (4, 5, 6) | -4 < y < 4 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, 2:4, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 6, size=(3,)),
            ),
            id="x[1, 2:4, y] where x.shape == (4, 5, 6) | 0 < y < 6 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, 2:4, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-6, 6, size=(3,)),
            ),
            id="x[1, 2:4, y] where x.shape == (4, 5, 6) | -6 < y < 6 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[0:2, 3, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 6, size=(3,)),
            ),
            id="x[0:2, 3, y] where x.shape == (4, 5, 6) | 0 < y < 6 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[0:2, 3, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-6, 6, size=(3,)),
            ),
            id="x[0:2, 3, y] where x.shape == (4, 5, 6) | -6 < y < 6 where y.shape == (3,)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[[3, 1], y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 5),
            ),
            id="x[[3, 1], y, 2:4] where x.shape == (4, 5, 6) | 0 < y < 5",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[[3, 1], y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-5, 5),
            ),
            id="x[[3, 1], y, 2:4] where x.shape == (4, 5, 6) | -5 < y < 5",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, [4, 2]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 5),
            ),
            id="x[1:3, y, [4, 2]] where x.shape == (4, 5, 6) | 0 < y < 5",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, [4, 2]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-5, 5),
            ),
            id="x[1:3, y, [4, 2]] where x.shape == (4, 5, 6) | -5 < y < 5",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, [3, 1], 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 4),
            ),
            id="x[y, [3, 1], 2:4] where x.shape == (4, 5, 6) | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, [3, 1], 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-4, 4),
            ),
            id="x[y, [3, 1], 2:4] where x.shape == (4, 5, 6) | -4 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1:3, [4, 2]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 4),
            ),
            id="x[y, 1:3, [4, 2]] where x.shape == (4, 5, 6) | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1:3, [4, 2]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-4, 4),
            ),
            id="x[y, 1:3, [4, 2]] where x.shape == (4, 5, 6) | -4 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[[1, 0], 2:4, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 6),
            ),
            id="x[[1, 0], 2:4, y] where x.shape == (4, 5, 6) | 0 < y < 6",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[[1, 0], 2:4, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-6, 6),
            ),
            id="x[[1, 0], 2:4, y] where x.shape == (4, 5, 6) | -6 < y < 6",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[0:2, [4, 2], y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 6),
            ),
            id="x[0:2, [4, 2], y] where x.shape == (4, 5, 6) | 0 < y < 6",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[0:2, [4, 2], y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-6, 6),
            ),
            id="x[0:2, [4, 2], y] where x.shape == (4, 5, 6) | -6 < y < 6",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[3, y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 5, size=(3, 2)),
            ),
            id="x[3, y, 2:4] where x.shape == (4, 5, 6) | 0 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[3, y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-5, 5, size=(3, 2)),
            ),
            id="x[3, y, 2:4] where x.shape == (4, 5, 6) | -5 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, 4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 5, size=(3, 2)),
            ),
            id="x[1:3, y, 4] where x.shape == (4, 5, 6) | 0 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, 4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-5, 5, size=(3, 2)),
            ),
            id="x[1:3, y, 4] where x.shape == (4, 5, 6) | -5 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 3, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 4, size=(3, 2)),
            ),
            id="x[y, 3, 2:4] where x.shape == (4, 5, 6) | 0 < y < 4 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 3, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-4, 4, size=(3, 2)),
            ),
            id="x[y, 3, 2:4] where x.shape == (4, 5, 6) | -4 < y < 4 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1:3, 4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 4, size=(3, 2)),
            ),
            id="x[y, 1:3, 4] where x.shape == (4, 5, 6) | 0 < y < 4 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1:3, 4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-4, 4, size=(3, 2)),
            ),
            id="x[y, 1:3, 4] where x.shape == (4, 5, 6) | -4 < y < 4 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, 2:4, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 6, size=(3, 2)),
            ),
            id="x[1, 2:4, y] where x.shape == (4, 5, 6) | 0 < y < 6 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, 2:4, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-6, 6, size=(3, 2)),
            ),
            id="x[1, 2:4, y] where x.shape == (4, 5, 6) | -6 < y < 6 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[0:2, 3, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 6, size=(3, 2)),
            ),
            id="x[0:2, 3, y] where x.shape == (4, 5, 6) | 0 < y < 6 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[0:2, 3, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-6, 6, size=(3, 2)),
            ),
            id="x[0:2, 3, y] where x.shape == (4, 5, 6) | -6 < y < 6 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[[3, 1], y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 5, size=(3, 2)),
            ),
            id="x[[3, 1], y, 2:4] where x.shape == (4, 5, 6) | 0 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[[3, 1], y, 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-5, 5, size=(3, 2)),
            ),
            id="x[[3, 1], y, 2:4] where x.shape == (4, 5, 6) | -5 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, [4, 2]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 5, size=(3, 2)),
            ),
            id="x[1:3, y, [4, 2]] where x.shape == (4, 5, 6) | 0 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1:3, y, [4, 2]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-5, 5, size=(3, 2)),
            ),
            id="x[1:3, y, [4, 2]] where x.shape == (4, 5, 6) | -5 < y < 5 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, [3, 1], 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 4, size=(3, 2)),
            ),
            id="x[y, [3, 1], 2:4] where x.shape == (4, 5, 6) | 0 < y < 4 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, [3, 1], 2:4],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-4, 4, size=(3, 2)),
            ),
            id="x[y, [3, 1], 2:4] where x.shape == (4, 5, 6) | -4 < y < 4 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1:3, [4, 2]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 4, size=(3, 2)),
            ),
            id="x[y, 1:3, [4, 2]] where x.shape == (4, 5, 6) | 0 < y < 4 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 1:3, [4, 2]],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-4, 4, size=(3, 2)),
            ),
            id="x[y, 1:3, [4, 2]] where x.shape == (4, 5, 6) | -4 < y < 4 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[[1, 0], 2:4, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 6, size=(3, 2)),
            ),
            id="x[[1, 0], 2:4, y] where x.shape == (4, 5, 6) | 0 < y < 6 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[[1, 0], 2:4, y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-6, 6, size=(3, 2)),
            ),
            id="x[[1, 0], 2:4, y] where x.shape == (4, 5, 6) | -6 < y < 6 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[0:2, [4, 2], y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(0, 6, size=(3, 2)),
            ),
            id="x[0:2, [4, 2], y] where x.shape == (4, 5, 6) | 0 < y < 6 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[0:2, [4, 2], y],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 4, 5, 6],
                lambda _: np.random.randint(-6, 6, size=(3, 2)),
            ),
            id="x[0:2, [4, 2], y] where x.shape == (4, 5, 6) | -6 < y < 6 where y.shape == (3, 2)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[:, y, :],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 10, 20, 30],
                lambda _: np.random.randint(0, 20, size=(2, 3, 4)),
            ),
            id="x[:, y, :] where x.shape == (10, 20, 30) | 0 < y < 20 where y.shape == (2, 3, 4)",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[:, y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 4, size=(2, 3, 4)),
                lambda _: np.random.randint(0, 5, size=(3, 4)),
            ),
            id=(
                "x[:, y, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 4 where y.shape == (2, 3, 4) "
                "| 0 < z < 5 where z.shape == (3, 4)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[:, y, :, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 4, size=(2, 3, 4)),
                lambda _: np.random.randint(0, 6, size=(3, 4)),
            ),
            id=(
                "x[:, y, :, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 4 where y.shape == (2, 3, 4) "
                "| 0 < z < 6 where z.shape == (3, 4)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, :, :, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 3, size=(2, 3, 4)),
                lambda _: np.random.randint(0, 6, size=(3, 4)),
            ),
            id=(
                "x[y, :, :, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 3 where y.shape == (2, 3, 4) "
                "| 0 < z < 6 where z.shape == (3, 4)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, 2:, [3, 1, 0, 4], :3, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 3, size=(2, 3, 4)),
                lambda _: np.random.randint(0, 7, size=(3, 4)),
            ),
            id=(
                "x[y, 2:, [3, 1, 0, 4], :3, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 3 where y.shape == (2, 3, 4) "
                "| 0 < z < 7 where z.shape == (3, 4)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[1:, y, [3, 1, 0, 4], :3, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 4, size=(2, 3, 4)),
                lambda _: np.random.randint(0, 7, size=(3, 4)),
            ),
            id=(
                "x[1:, y, [3, 1, 0, 4], :3, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 4 where y.shape == (2, 3, 4) "
                "| 0 < z < 7 where z.shape == (3, 4)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, 2:, [3, 1, 0, 4], :3, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 3),
                lambda _: np.random.randint(0, 7, size=(3, 4)),
            ),
            id=(
                "x[y, 2:, [3, 1, 0, 4], :3, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 3 "
                "| 0 < z < 7 where z.shape == (3, 4)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[1:, y, [3, 1, 0, 4], :3, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 4),
                lambda _: np.random.randint(0, 7, size=(3, 4)),
            ),
            id=(
                "x[1:, y, [3, 1, 0, 4], :3, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 4 "
                "| 0 < z < 7 where z.shape == (3, 4)"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, 2:, [3, 1, 0, 4], :3, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 3, size=(2, 3, 4)),
                lambda _: np.random.randint(0, 7),
            ),
            id=(
                "x[y, 2:, [3, 1, 0, 4], :3, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 3 where y.shape == (2, 3, 4) "
                "| 0 < z < 7"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[1:, y, [3, 1, 0, 4], :3, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 4, size=(2, 3, 4)),
                lambda _: np.random.randint(0, 7),
            ),
            id=(
                "x[1:, y, [3, 1, 0, 4], :3, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 4 where y.shape == (2, 3, 4) "
                "| 0 < z < 7 "
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, 2:, [3, 1, 0, 4], :3, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 3),
                lambda _: np.random.randint(0, 7),
            ),
            id=(
                "x[y, 2:, [3, 1, 0, 4], :3, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 3 "
                "| 0 < z < 7"
            ),
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[1:, y, [3, 1, 0, 4], :3, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4, 5, 6, 7],
                lambda _: np.random.randint(0, 4),
                lambda _: np.random.randint(0, 7),
            ),
            id=(
                "x[1:, y, [3, 1, 0, 4], :3, z] where x.shape == (3, 4, 5, 6, 7) "
                "| 0 < y < 4 "
                "| 0 < z < 7 "
            ),
        ),
    ],
)
def test_dynamic_indexing(encryption_status, function, inputset, helpers):
    """
    Test dynamic indexing.
    """

    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, encryption_status)
    circuit = compiler.compile(inputset, configuration)

    for sample in random.sample(inputset, 8):
        helpers.check_execution(circuit, function, list(sample))
