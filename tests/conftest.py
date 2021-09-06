"""PyTest configuration file"""
from typing import Callable, Dict, Type

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest

from concrete.common.representation.intermediate import (
    ALL_IR_NODES,
    Add,
    ArbitraryFunction,
    Constant,
    Dot,
    Input,
    IntermediateNode,
    Mul,
    Sub,
)


def _is_equivalent_to_binary_commutative(lhs: IntermediateNode, rhs: object) -> bool:
    """is_equivalent_to for a binary and commutative operation."""
    return (
        isinstance(rhs, lhs.__class__)
        and (lhs.inputs == rhs.inputs or lhs.inputs == rhs.inputs[::-1])
        and lhs.outputs == rhs.outputs
    )


def _is_equivalent_to_binary_non_commutative(lhs: IntermediateNode, rhs: object) -> bool:
    """is_equivalent_to for a binary and non-commutative operation."""
    return (
        isinstance(rhs, lhs.__class__) and lhs.inputs == rhs.inputs and lhs.outputs == rhs.outputs
    )


def is_equivalent_add(lhs: Add, rhs: object) -> bool:
    """Helper function to check if an Add node is equivalent to an other object."""
    return _is_equivalent_to_binary_commutative(lhs, rhs)


def is_equivalent_arbitrary_function(lhs: ArbitraryFunction, rhs: object) -> bool:
    """Helper function to check if an ArbitraryFunction node is equivalent to an other object."""
    return (
        isinstance(rhs, ArbitraryFunction)
        and lhs.op_args == rhs.op_args
        and lhs.op_kwargs == rhs.op_kwargs
        and lhs.op_name == rhs.op_name
        and is_equivalent_intermediate_node(lhs, rhs)
    )


def is_equivalent_constant(lhs: Constant, rhs: object) -> bool:
    """Helper function to check if a Constant node is equivalent to an other object."""
    return (
        isinstance(rhs, Constant)
        and lhs.constant_data == rhs.constant_data
        and is_equivalent_intermediate_node(lhs, rhs)
    )


def is_equivalent_dot(lhs: Dot, rhs: object) -> bool:
    """Helper function to check if a Dot node is equivalent to an other object."""
    return (
        isinstance(rhs, Dot)
        and lhs.evaluation_function == rhs.evaluation_function
        and is_equivalent_intermediate_node(lhs, rhs)
    )


def is_equivalent_input(lhs: Input, rhs: object) -> bool:
    """Helper function to check if an Input node is equivalent to an other object."""
    return (
        isinstance(rhs, Input)
        and lhs.input_name == rhs.input_name
        and lhs.program_input_idx == rhs.program_input_idx
        and is_equivalent_intermediate_node(lhs, rhs)
    )


def is_equivalent_mul(lhs: Mul, rhs: object) -> bool:
    """Helper function to check if a Mul node is equivalent to an other object."""
    return _is_equivalent_to_binary_commutative(lhs, rhs)


def is_equivalent_sub(lhs: Sub, rhs: object) -> bool:
    """Helper function to check if a Sub node is equivalent to an other object."""
    return _is_equivalent_to_binary_non_commutative(lhs, rhs)


def is_equivalent_intermediate_node(lhs: IntermediateNode, rhs: object) -> bool:
    """Helper function to check if an IntermediateNode node is equivalent to an other object."""
    return (
        isinstance(rhs, IntermediateNode)
        and lhs.inputs == rhs.inputs
        and lhs.outputs == rhs.outputs
    )


EQUIVALENT_TEST_FUNC: Dict[Type, Callable[..., bool]] = {
    Add: is_equivalent_add,
    ArbitraryFunction: is_equivalent_arbitrary_function,
    Constant: is_equivalent_constant,
    Dot: is_equivalent_dot,
    Input: is_equivalent_input,
    Mul: is_equivalent_mul,
    Sub: is_equivalent_sub,
}

_missing_nodes_in_mapping = ALL_IR_NODES - EQUIVALENT_TEST_FUNC.keys()
assert len(_missing_nodes_in_mapping) == 0, (
    f"Missing IR node in EQUIVALENT_TEST_FUNC : "
    f"{', '.join(sorted(str(node_type) for node_type in _missing_nodes_in_mapping))}"
)

del _missing_nodes_in_mapping


class TestHelpers:
    """Class allowing to pass helper functions to tests"""

    @staticmethod
    def nodes_are_equivalent(lhs, rhs) -> bool:
        """Helper function for tests to check if two nodes are equivalent."""
        equivalent_func = EQUIVALENT_TEST_FUNC.get(type(lhs), None)
        if equivalent_func is not None:
            return equivalent_func(lhs, rhs)

        # This is a default for the test_conftest.py that should remain separate from the package
        # nodes is_equivalent_* functions
        return lhs.is_equivalent_to(rhs)

    @staticmethod
    def digraphs_are_equivalent(reference: nx.MultiDiGraph, to_compare: nx.MultiDiGraph):
        """Check that two digraphs are equivalent without modifications"""
        # edge_match is a copy of node_match
        edge_matcher = iso.categorical_multiedge_match("input_idx", None)
        node_matcher = iso.generic_node_match(
            "_test_content", None, TestHelpers.nodes_are_equivalent
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
