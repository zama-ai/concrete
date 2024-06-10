"""
Declaration of classes related to composition.
"""

# pylint: disable=import-error,no-name-in-module

from typing import Iterable, List, NamedTuple, Protocol, Tuple, runtime_checkable

from ..representation import Graph


class CompositionClause(NamedTuple):
    """
    A raw composition clause.
    """

    func: str
    pos: int

    @staticmethod
    def create(tup: Tuple[str, int]) -> "CompositionClause":
        """
        Create a composition clause from a tuple of a function name and a position.
        """
        return CompositionClause(tup[0], tup[1])


class CompositionRule(NamedTuple):
    """
    A raw composition rule.
    """

    from_: CompositionClause
    to: CompositionClause

    @staticmethod
    def create(tup: Tuple[CompositionClause, CompositionClause]) -> "CompositionRule":
        """
        Create a composition rule from a tuple containing an output clause and an input clause.
        """
        return CompositionRule(tup[0], tup[1])


@runtime_checkable
class CompositionPolicy(Protocol):
    """
    A protocol for composition policies.
    """

    def get_rules_iter(self, funcs: List[Graph]) -> Iterable[CompositionRule]:
        """
        Return an iterator over composition rules.
        """
