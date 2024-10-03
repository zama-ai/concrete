"""Construct a graphviz graph from tlu nodes."""


def detect_std_lut_label(node):
    """Give friendly name to some known tlus."""
    inputs = node.arguments
    content = node.content
    outputs = node.results
    assert len(outputs) == 1
    if len(inputs) == 1:
        return "l1-not" if content == [1, 0] else "l1-id"
    if len(inputs) == 2:
        if content == [[0, 0], [0, 1]]:
            return "l2-carry"
        if content == [[1, 1], [1, 0]]:
            return "l2-nand"
        if content == [[0, 1], [1, 1]]:
            return "l2-or"
        if content == [[1, 0], [0, 0]]:
            return "l2-nor"
        if content == [[0, 1], [1, 0]]:
            return "l2-add"
        if content == [[1, 0], [0, 1]]:
            return "l2-nadd"
    elif len(inputs) == 3:
        if content == [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]:
            return "l3-add"
        if content == [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]:
            return "l3-add(n,_,_)"
        if content == [[[1, 0], [0, 0]], [[1, 1], [1, 0]]]:
            return "l3-ncarry(n,_,_)"
        if content == [[[1, 1], [1, 0]], [[1, 0], [0, 0]]]:
            return "l3-ncarry(_,_,_)"
    elif len(inputs) == 4:
        if content == [[[[1, 0], [0, 1]], [[0, 1], [0, 1]]], [[[1, 0], [1, 0]], [[1, 0], [0, 1]]]]:
            return "l4-add(n,_,n,_)"
    return None


# def _detect_std_lut_label(node):
#     inputs = node.arguments
#     content = node.content
#     outputs = node.results
#     assert len(outputs) == 1
#     if len(inputs) == 0:
#         return f"c-{content[0]}"

#     if len(inputs) == 1:
#         if content == [1, 0]:
#             return "l1-not"
#         else:
#             return "l1-id"
#     if len(inputs) == 2:
#         if content == [[0, 0], [0, 1]]:
#             return "l2-carry"
#         if content == [[1, 1], [1, 0]]:
#             return "l2-nand"
#         if content == [[0, 1], [1, 1]]:
#             return "l2-or"
#         if content == [[1, 0], [0, 0]]:
#             return "l2-nor"
#         if content == [[0, 1], [1, 0]]:
#             return "l2-add"
#         if content == [[1, 0], [0, 1]]:
#             return "l2-add"
#     elif len(inputs) == 3:
#         if content ==  [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]:
#             return "l3-add"
#         elif content == [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]:
#             return "l3-add"
#         elif content == [[[1, 0], [0, 0]], [[1, 1], [1, 0]]]:
#             return "l3-carry"
#         elif content == [[[1, 1], [1, 0]], [[1, 0], [0, 0]]]:
#             return "l3-carry"
#     elif len(inputs) == 4:
#         if [[[[1, 0], [0, 1]], [[0, 1], [0, 1]]], [[[1, 0], [1, 0]], [[1, 0], [0, 1]]]]:
#             return "l4-add"
#     return None


def to_graph(name, nodes):
    """Construct a graphviz graph from tlu nodes."""
    try:
        import graphviz  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        msg = "You must install concrete-python with graphviz support or install graphviz manually"
        raise ValueError(msg) from exc

    graph = graphviz.Digraph(name=name)
    declated_node = set()
    with graph.subgraph(name="cluster_output") as cluster:
        for tlu in nodes:
            outputs = tlu.results
            for result in outputs:
                if result.is_interface:
                    cluster.node(result.name, result.name, color="magenta", shape="box")
                    declated_node.add(result.name)
    with graph.subgraph(name="cluster_input") as cluster:
        for tlu in nodes:
            inputs = tlu.arguments
            for argument in sorted(inputs, key=lambda n: n.name):
                if argument.is_interface and argument.name not in declated_node:
                    cluster.node(argument.name, argument.name, color="blue", shape="box")
                    declated_node.add(argument.name)

    def node_name_label(node, i):
        label = detect_std_lut_label(node) or f"l-{len(node.arguments)}"
        name = f"{label}-{i}"
        return name, node.name + " = " + label

    with graph.subgraph(name="cluster_inner") as cluster:
        for i, tlu in enumerate(nodes):
            inputs = tlu.arguments
            outputs = tlu.results
            name, label = node_name_label(tlu, i)
            cluster.node(name, label, shape="octagon")
            for argument in inputs:
                if argument.name not in declated_node:
                    cluster.node(argument.name, argument.name, color="black", shape="point")
                    declated_node.add(argument.name)
            for result in outputs:
                if result.name not in declated_node:
                    cluster.node(result.name, result.name, color="black", shape="point")
                    declated_node.add(result.name)
    for i, tlu in enumerate(nodes):
        inputs = tlu.arguments
        outputs = tlu.results
        name, label = node_name_label(tlu, i)
        for j, argument in enumerate(inputs):
            assert argument.name in declated_node
            graph.edge(argument.name, name, headlabel=str(j))
        for result in outputs:
            assert result.name in declated_node
            graph.edge(name, result.name)
    return graph
