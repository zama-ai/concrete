"""Test file for HDK's hnumpy debugging functions"""

import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import ClearValue, EncryptedValue
from hdk.common.debugging import draw_graph
from hdk.hnumpy import tracing


@pytest.mark.parametrize(
    "lambda_f",
    [
        lambda x, y: x + y,
        lambda x, y: x + x - y * y * y + x,
    ],
)
@pytest.mark.parametrize(
    "x",
    [
        pytest.param(EncryptedValue(Integer(64, is_signed=False)), id="Encrypted uint"),
        # pytest.param(
        #     EncryptedValue(Integer(64, is_signed=True)),
        #     id="Encrypted int",
        # ),
        # pytest.param(
        #     ClearValue(Integer(64, is_signed=False)),
        #     id="Clear uint",
        # ),
        # pytest.param(
        #     ClearValue(Integer(64, is_signed=True)),
        #     id="Clear int",
        # ),
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        pytest.param(EncryptedValue(Integer(64, is_signed=False)), id="Encrypted uint"),
        # pytest.param(
        #     EncryptedValue(Integer(64, is_signed=True)),
        #     id="Encrypted int",
        # ),
        pytest.param(
            ClearValue(Integer(64, is_signed=False)),
            id="Clear uint",
        ),
        # pytest.param(
        #     ClearValue(Integer(64, is_signed=True)),
        #     id="Clear int",
        # ),
    ],
)
def test_hnumpy_draw_graph(lambda_f, x, y):
    "Test hnumpy draw_graph"
    graph = tracing.trace_numpy_function(lambda_f, {"x": x, "y": y})

    draw_graph(graph, block_until_user_closes_graph=False)
