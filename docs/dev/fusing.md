# Fusing

Fusing is the act of combining multiple nodes into a single node, which is converted to a table lookup.

## How is it done?

Code related to fusing is in the `concrete/numpy/compilation/utils.py` file.

Fusing can be performed using the `fuse` function. Within `fuse`:

1. We loop until there are no more subgraphs to fuse.
2. Within each iteration:
3.  We find a subgraph to fuse.

    3.1.  We search for a terminal node that is appropriate for fusing.

    3.2. We crawl backwards to find the closest integer nodes to this node.

    3.3.  If there is a single node as such, we return the subgraph from this node to the terminal node.

    3.4. Otherwise, we try to find the lowest common ancestor (lca) of this list of nodes.

    3.5. If lca doesn't exist, we say this particular terminal node is not fusable, and we go back to search for another subgraph.

    3.6. Otherwise, we use this lca as the input of the subgraph and continue with `subgraph` node creation below.
4.  We convert the subgraph into a `subgraph` node

    4.1.  We check fusability status of the nodes of the subgraph in this step.
5. We substitute the `subgraph` node to the original graph.

## Limitations

With the current implementation, we cannot fuse subgraphs that depend on multiple encrypted values where those values doesn't have a common lca (e.g., `np.round(np.sin(x) + np.cos(y))`).

{% hint style="info" %}
[Kolmogorov–Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold\_representation\_theorem) states that every multivariate continuous function can be represented as a superposition of continuous functions of one variable. Therefore, the case above could be handled in future versions of **Concrete Numpy**.
{% endhint %}
