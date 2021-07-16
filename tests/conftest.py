"""PyTest configuration file"""
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest


class TestHelpers:
    """Class allowing to pass helper functions to tests"""

    @staticmethod
    def digraphs_are_equivalent(reference: nx.DiGraph, to_compare: nx.DiGraph):
        """Check that two digraphs are equivalent without modifications"""
        # edge_match is a copy of node_match
        edge_matcher = iso.categorical_node_match("input_idx", None)
        node_matcher = iso.categorical_node_match("content", None)
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
