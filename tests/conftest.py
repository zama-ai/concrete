"""PyTest configuration file"""
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest


class TestHelpers:
    """Class allowing to pass helper functions to tests"""

    @staticmethod
    def digraphs_are_equivalent(reference: nx.MultiDiGraph, to_compare: nx.MultiDiGraph):
        """Check that two digraphs are equivalent without modifications"""
        # edge_match is a copy of node_match
        edge_matcher = iso.categorical_multiedge_match("input_idx", None)
        node_matcher = iso.generic_node_match(
            "_test_content", None, lambda lhs, rhs: lhs.is_equivalent_to(rhs)
        )

        # Set the _test_content for each node in the graphs
        for node in reference.nodes():
            reference.add_node(node, _test_content=node)

        for node in to_compare.nodes():
            to_compare.add_node(node, _test_content=node)

        graphs_are_isomorphic = nx.is_isomorphic(
            reference,
            to_compare,
            node_match=node_matcher,
            edge_match=edge_matcher,
        )

        return graphs_are_isomorphic


@pytest.fixture
def test_helpers():
    """Fixture to return the static helper class"""
    return TestHelpers
