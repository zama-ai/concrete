"""Test file for drawing"""

import tempfile
from pathlib import Path

from hdk.common.data_types.integers import Integer
from hdk.common.debugging import draw_graph
from hdk.common.values import EncryptedScalar
from hdk.hnumpy.compile import compile_numpy_function_into_op_graph


def test_draw_graph_with_saving():
    """Tests drawing and saving a graph"""

    def function(x):
        return x + 42

    op_graph = compile_numpy_function_into_op_graph(
        function,
        {"x": EncryptedScalar(Integer(7, True))},
        iter([(-2,), (-1,), (0,), (1,), (2,)]),
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_directory = Path(tmp)
        output_file = output_directory.joinpath("test.png")
        draw_graph(op_graph, save_to=output_file)
        assert output_file.exists()
