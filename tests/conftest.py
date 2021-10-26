"""PyTest configuration file"""
import json
import operator
import re
from pathlib import Path
from typing import Callable, Dict, Type

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest

from concrete.common.compilation import CompilationConfiguration
from concrete.common.representation.intermediate import (
    ALL_IR_NODES,
    Add,
    Constant,
    Dot,
    GenericFunction,
    IndexConstant,
    Input,
    IntermediateNode,
    MatMul,
    Mul,
    Sub,
    UnivariateFunction,
)


def pytest_addoption(parser):
    """Options for pytest"""

    parser.addoption(
        "--global-coverage-infos-json",
        action="store",
        default=None,
        type=str,
        help="To dump pytest-cov term report to a text file.",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus):  # pylint: disable=unused-argument
    """Pytest callback when testing ends."""
    # Hacked together from the source code, they don't have an option to export to file and it's too
    # much work to get a PR in for such a little thing
    # https://github.com/pytest-dev/pytest-cov/blob/
    # ec344d8adf2d78238d8f07cb20ed2463d7536970/src/pytest_cov/plugin.py#L329
    if session.config.pluginmanager.hasplugin("_cov"):
        global_coverage_file = session.config.getoption(
            "--global-coverage-infos-json", default=None
        )
        if global_coverage_file is not None:
            cov_plugin = session.config.pluginmanager.getplugin("_cov")
            coverage_txt = cov_plugin.cov_report.getvalue()
            coverage_status = 0
            if (
                cov_plugin.options.cov_fail_under is not None
                and cov_plugin.options.cov_fail_under > 0
            ):
                failed = cov_plugin.cov_total < cov_plugin.options.cov_fail_under
                # If failed is False coverage_status is 0, if True it's 1
                coverage_status = int(failed)
            global_coverage_file_path = Path(global_coverage_file).resolve()
            with open(global_coverage_file_path, "w", encoding="utf-8") as f:
                json.dump({"exit_code": coverage_status, "content": coverage_txt}, f)


def _is_equivalent_to_binary_commutative(lhs: IntermediateNode, rhs: object) -> bool:
    """is_equivalent_to for a binary and commutative operation."""
    return (
        isinstance(rhs, lhs.__class__)
        and (lhs.inputs in (rhs.inputs, rhs.inputs[::-1]))
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


# From https://stackoverflow.com/a/28635464
_code_and_constants_attr_getter = operator.attrgetter("co_code", "co_consts")


def _code_and_constants(object_):
    """Helper function to get python code and constants"""
    return _code_and_constants_attr_getter(object_.__code__)


def python_functions_are_equal_or_equivalent(lhs: object, rhs: object) -> bool:
    """Helper function to check if two functions are equal or their code are equivalent.

    This is not perfect, but will be good enough for tests.
    """

    if lhs == rhs:
        return True

    try:
        lhs_code_and_constants = _code_and_constants(lhs)
        rhs_code_and_constants = _code_and_constants(rhs)
        return lhs_code_and_constants == rhs_code_and_constants
    except AttributeError:
        return False


def is_equivalent_arbitrary_function(lhs: UnivariateFunction, rhs: object) -> bool:
    """Helper function to check if an UnivariateFunction node is equivalent to an other object."""
    return (
        isinstance(rhs, UnivariateFunction)
        and python_functions_are_equal_or_equivalent(lhs.arbitrary_func, rhs.arbitrary_func)
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


def is_equivalent_index_constant(lhs: IndexConstant, rhs: object) -> bool:
    """Helper function to check if an IndexConstant node is equivalent to an other object."""
    return (
        isinstance(rhs, IndexConstant)
        and lhs.index == rhs.index
        and is_equivalent_intermediate_node(lhs, rhs)
    )


def is_equivalent_mul(lhs: Mul, rhs: object) -> bool:
    """Helper function to check if a Mul node is equivalent to an other object."""
    return _is_equivalent_to_binary_commutative(lhs, rhs)


def is_equivalent_sub(lhs: Sub, rhs: object) -> bool:
    """Helper function to check if a Sub node is equivalent to an other object."""
    return _is_equivalent_to_binary_non_commutative(lhs, rhs)


def is_equivalent_matmul(lhs: MatMul, rhs: object) -> bool:
    """Helper function to check if a MatMul node is equivalent to an other object."""
    return isinstance(rhs, MatMul) and is_equivalent_intermediate_node(lhs, rhs)


def is_equivalent_intermediate_node(lhs: IntermediateNode, rhs: object) -> bool:
    """Helper function to check if an IntermediateNode node is equivalent to an other object."""
    return (
        isinstance(rhs, IntermediateNode)
        and lhs.inputs == rhs.inputs
        and lhs.outputs == rhs.outputs
    )


EQUIVALENT_TEST_FUNC: Dict[Type, Callable[..., bool]] = {
    Add: is_equivalent_add,
    UnivariateFunction: is_equivalent_arbitrary_function,
    GenericFunction: is_equivalent_arbitrary_function,
    Constant: is_equivalent_constant,
    Dot: is_equivalent_dot,
    IndexConstant: is_equivalent_index_constant,
    Input: is_equivalent_input,
    Mul: is_equivalent_mul,
    Sub: is_equivalent_sub,
    MatMul: is_equivalent_matmul,
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
        edge_matcher = iso.categorical_multiedge_match(["input_idx", "output_idx"], [None, None])
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

    @staticmethod
    def python_functions_are_equal_or_equivalent(lhs, rhs):
        """Helper function to check if two functions are equal or their code are equivalent.

        This is not perfect, but will be good enough for tests.
        """
        return python_functions_are_equal_or_equivalent(lhs, rhs)


@pytest.fixture
def test_helpers():
    """Fixture to return the static helper class"""
    return TestHelpers


@pytest.fixture
def default_compilation_configuration():
    """Return the default test compilation configuration"""
    return CompilationConfiguration(
        dump_artifacts_on_unexpected_failures=False,
        treat_warnings_as_errors=True,
    )


REMOVE_COLOR_CODES_RE = re.compile(r"\x1b[^m]*m")


@pytest.fixture
def remove_color_codes():
    """Return the re object to remove color codes"""
    return lambda x: REMOVE_COLOR_CODES_RE.sub("", x)
