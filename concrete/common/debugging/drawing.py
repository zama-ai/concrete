"""functions to draw the different graphs we can generate in the package, eg to debug."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from ..debugging.custom_assert import assert_true
from ..operator_graph import OPGraph
from ..representation.intermediate import (
    ALL_IR_NODES,
    Add,
    Constant,
    Conv2D,
    Dot,
    GenericFunction,
    IndexConstant,
    Input,
    MatMul,
    Mul,
    Sub,
)

IR_NODE_COLOR_MAPPING = {
    Input: "blue",
    Constant: "cyan",
    Conv2D: "brown",
    Add: "red",
    Sub: "yellow",
    Mul: "green",
    GenericFunction: "orange",
    IndexConstant: "black",
    Dot: "purple",
    MatMul: "brown",
    "GenericFunction": "orange",
    "TLU": "grey",
    "output": "magenta",
}

_missing_nodes_in_mapping = ALL_IR_NODES - IR_NODE_COLOR_MAPPING.keys()
assert_true(
    len(_missing_nodes_in_mapping) == 0,
    (
        f"Missing IR node in IR_NODE_COLOR_MAPPING : "
        f"{', '.join(sorted(str(node_type) for node_type in _missing_nodes_in_mapping))}"
    ),
)

del _missing_nodes_in_mapping


def draw_graph(
    op_graph: OPGraph,
    show: bool = False,
    vertical: bool = True,
    save_to: Optional[Path] = None,
) -> str:
    """Draws operation graphs and optionally saves/shows the drawing.

    Note that this function requires the python `pygraphviz` package which itself requires the
    installation of `graphviz` packages, see
    https://pygraphviz.github.io/documentation/stable/install.html

    Args:
        op_graph (OPGraph): the operation graph to be drawn and optionally saved/shown
        show (bool): if set to True, the drawing will be shown using matplotlib
        vertical (bool): if set to True, the orientation will be vertical
        save_to (Optional[Path]): if specified, the drawn graph will be saved to this path; else
            it is saved in a temporary file

    Returns:
        The path of the file where the drawn graph is saved

    """

    def get_color(node, output_nodes):
        value_to_return = IR_NODE_COLOR_MAPPING[type(node)]
        if node in output_nodes:
            value_to_return = IR_NODE_COLOR_MAPPING["output"]
        elif isinstance(node, GenericFunction):
            value_to_return = IR_NODE_COLOR_MAPPING.get(node.op_name, value_to_return)
        return value_to_return

    graph = op_graph.graph
    output_nodes = set(op_graph.output_nodes.values())

    attributes = {
        node: {
            "label": node.text_for_drawing(),
            "color": get_color(node, output_nodes),
            "penwidth": 2,  # double thickness for circles
            "peripheries": 2 if node in output_nodes else 1,  # double circle for output nodes
        }
        for node in graph.nodes
    }
    nx.set_node_attributes(graph, attributes)

    # TODO: #639 adapt drawing routine to manage output_idx
    for edge in graph.edges(keys=True):
        idx = graph.edges[edge]["input_idx"]
        graph.edges[edge]["label"] = f" {idx} "  # spaces are there intentionally for a better look

    try:
        agraph = nx.nx_agraph.to_agraph(graph)
    except ImportError as e:  # pragma: no cover
        if "pygraphviz" in str(e):
            err_msg = (
                f"{draw_graph.__name__} requires pygraphviz, install your OS graphviz distribution "
                "https://pygraphviz.github.io/documentation/stable/install.html "
                "and reinstall with extras: `pip install --force-reinstall "
                "concrete-numpy[full]`"
            )
            raise ImportError(err_msg) from e
    agraph.graph_attr["rankdir"] = "TB" if vertical else "LR"
    agraph.layout("dot")

    if save_to is None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # we need to change the permissions of the temporary file
            # so that it can be read by all users

            # (https://stackoverflow.com/a/44130605)

            # get the old umask and replace it with 0o666
            old_umask = os.umask(0o666)

            # restore the old umask back
            os.umask(old_umask)

            # combine the old umask with the wanted permissions
            permissions = 0o666 & ~old_umask

            # set new permissions
            os.chmod(tmp.name, permissions)

            save_to_str = str(tmp.name)
    else:
        save_to_str = str(save_to)

    agraph.draw(save_to_str)

    if show:  # pragma: no cover
        # We can't have coverage in this branch as `plt.show()` blocks and waits for user action.
        plt.close("all")
        plt.figure()
        img = Image.open(save_to_str)
        plt.imshow(img)
        img.close()
        plt.axis("off")
        plt.show()

    return save_to_str
