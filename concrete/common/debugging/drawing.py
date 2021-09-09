"""functions to draw the different graphs we can generate in the package, eg to debug."""

import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from ..debugging.custom_assert import custom_assert
from ..operator_graph import OPGraph
from ..representation import intermediate as ir
from ..representation.intermediate import ALL_IR_NODES

IR_NODE_COLOR_MAPPING = {
    ir.Input: "blue",
    ir.Constant: "cyan",
    ir.Add: "red",
    ir.Sub: "yellow",
    ir.Mul: "green",
    ir.ArbitraryFunction: "orange",
    ir.Dot: "purple",
    "ArbitraryFunction": "orange",
    "TLU": "grey",
    "output": "magenta",
}

_missing_nodes_in_mapping = ALL_IR_NODES - IR_NODE_COLOR_MAPPING.keys()
custom_assert(
    len(_missing_nodes_in_mapping) == 0,
    (
        f"Missing IR node in IR_NODE_COLOR_MAPPING : "
        f"{', '.join(sorted(str(node_type) for node_type in _missing_nodes_in_mapping))}"
    ),
)

del _missing_nodes_in_mapping


def draw_graph(
    opgraph: OPGraph,
    show: bool = False,
    vertical: bool = True,
    save_to: Optional[Path] = None,
) -> Image.Image:
    """Draws operation graphs and optionally saves/shows the drawing.

    Args:
        opgraph (OPGraph): the graph to be drawn and optionally saved/shown
        show (bool): if set to True, the drawing will be shown using matplotlib
        vertical (bool): if set to True, the orientation will be vertical
        save_to (Optional[Path]): if specified, the drawn graph will be saved to this path

    Returns:
        Pillow Image of the drawn graph.
        This is useful because you can use the drawing however you like.
        (check https://pillow.readthedocs.io/en/stable/reference/Image.html for further information)

    """

    def get_color(node, output_nodes):
        value_to_return = IR_NODE_COLOR_MAPPING[type(node)]
        if node in output_nodes:
            value_to_return = IR_NODE_COLOR_MAPPING["output"]
        elif isinstance(node, ir.ArbitraryFunction):
            value_to_return = IR_NODE_COLOR_MAPPING.get(node.op_name, value_to_return)
        return value_to_return

    graph = opgraph.graph
    output_nodes = set(opgraph.output_nodes.values())

    attributes = {
        node: {
            "label": node.label(),
            "color": get_color(node, output_nodes),
            "penwidth": 2,  # double thickness for circles
            "peripheries": 2 if node in output_nodes else 1,  # double circle for output nodes
        }
        for node in graph.nodes
    }
    nx.set_node_attributes(graph, attributes)

    for edge in graph.edges(keys=True):
        idx = graph.edges[edge]["input_idx"]
        graph.edges[edge]["label"] = f" {idx} "  # spaces are there intentionally for a better look

    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.graph_attr["rankdir"] = "TB" if vertical else "LR"
    agraph.layout("dot")

    if save_to is None:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            agraph.draw(tmp.name)
            img = Image.open(tmp.name)
    else:
        agraph.draw(save_to)
        img = Image.open(save_to)

    if show:  # pragma: no cover
        # We can't have coverage in this branch as `plt.show()` blocks and waits for user action.
        plt.close("all")
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    return img
