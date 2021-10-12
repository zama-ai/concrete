"""Test file for conftest helper functions"""
import networkx as nx


def test_digraphs_are_equivalent(test_helpers):
    """Function to test digraphs_are_equivalent helper function"""

    class TestNode:
        """Dummy test node"""

        computation: str

        def __init__(self, computation: str) -> None:
            self.computation = computation

        def __hash__(self) -> int:
            return self.computation.__hash__()

        def __eq__(self, other: object) -> bool:
            return isinstance(other, self.__class__) and self.computation == other.computation

        is_equivalent_to = __eq__

    g_1 = nx.MultiDiGraph()
    g_2 = nx.MultiDiGraph()

    t_0 = TestNode("Add")
    t_1 = TestNode("Mul")
    t_2 = TestNode("TLU")

    g_1.add_edge(t_0, t_2, input_idx=0, output_idx=0)
    g_1.add_edge(t_1, t_2, input_idx=1, output_idx=0)

    t0p = TestNode("Add")
    t1p = TestNode("Mul")
    t2p = TestNode("TLU")

    g_2.add_edge(t1p, t2p, input_idx=1, output_idx=0)
    g_2.add_edge(t0p, t2p, input_idx=0, output_idx=0)

    bad_g2 = nx.MultiDiGraph()

    bad_t0 = TestNode("Not Add")

    bad_g2.add_edge(bad_t0, t_2, input_idx=0, output_idx=0)
    bad_g2.add_edge(t_1, t_2, input_idx=1, output_idx=0)

    bad_g3 = nx.MultiDiGraph()

    bad_g3.add_edge(t_0, t_2, input_idx=1, output_idx=0)
    bad_g3.add_edge(t_1, t_2, input_idx=0, output_idx=0)

    assert test_helpers.digraphs_are_equivalent(g_1, g_2), "Graphs should be equivalent"
    assert not test_helpers.digraphs_are_equivalent(g_1, bad_g2), "Graphs should not be equivalent"
    assert not test_helpers.digraphs_are_equivalent(g_2, bad_g2), "Graphs should not be equivalent"
    assert not test_helpers.digraphs_are_equivalent(g_1, bad_g3), "Graphs should not be equivalent"
    assert not test_helpers.digraphs_are_equivalent(g_2, bad_g3), "Graphs should not be equivalent"
