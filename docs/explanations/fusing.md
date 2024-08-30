# Fusing

This document describes the concept of fusing, which is the act of combining multiple nodes into a single node, which is converted to a Table Lookup.

## How is it done?

Code related to fusing is in the `frontends/concrete-python/concrete/fhe/compilation/utils.py` file. Fusing can be performed using the `fuse` function.

Within `fuse`:

1. We loop until there are no more subgraphs to fuse.
2. <mark style="background-color:yellow;">Within each iteration:</mark>
    2.1. We find a subgraph to fuse.

    2.2. We search for a terminal node that is appropriate for fusing.

    2.3. We crawl backwards to find the closest integer nodes to this node.

    2.4. If there is a single node as such, we return the subgraph from this node to the terminal node.

    2.5. Otherwise, we try to find the lowest common ancestor (lca) of this list of nodes.

    2.6. If an lca doesn't exist, we say this particular terminal node is not fusable, and we go back to search for another subgraph.

    2.7. Otherwise, we use this lca as the input of the subgraph and continue with `subgraph` node creation below.

    2.8. We convert the subgraph into a `subgraph` node by checking fusability status of the nodes of the subgraph in this step.

    2.9. We substitute the `subgraph` node to the original graph.

## Limitations

With the current implementation, we cannot fuse subgraphs that depend on multiple encrypted values where those values don't have a common lca (e.g., `np.round(np.sin(x) + np.cos(y))`).
