"""
The path-between query: does anything in a given set of control-flow nodes lie on a path from one
node to another.

This is what a reaching-value question reduces to once dominance has ordered the two ends. The value
established at a definition still holds at a use when the definition runs first on every path — which
is dominance — and nothing that could change it runs in between. "In between" is the intersection of
what is reachable forward from the definition with what is reachable backward from the use, and a
kill outside that intersection cannot be on any path joining them.

What is a *kill* is entirely the caller's question and is not asked here: a language names the sites
at which the thing it tracks may change, or says it cannot enumerate them. The graph half is this
module's, and it is the same for every language.
"""
from __future__ import annotations

from typing import Iterable

from refinery.lib.scripts.analysis.cfg import CfgNode
from refinery.lib.scripts.analysis.dominance import DominatorModel


class ReachabilityQuery:
    """
    Memoized forward and backward reachability over one `DominatorModel`, and the path-between
    question asked from them.

    The memo is the reason this is an object rather than a function. One definition is asked against
    many uses, so the forward set from the definition is computed once and reused; the graphs do not
    change for as long as the model lives, so nothing invalidates it.
    """

    def __init__(self, dominators: DominatorModel):
        self._dominators = dominators
        self._reach: dict[tuple[int, bool], set[int]] = {}

    def reachable(self, node: CfgNode, *, forward: bool) -> set[int]:
        """
        The ids of the nodes reachable from *node* in the given direction, including *node*.
        """
        key = (id(node), forward)
        cached = self._reach.get(key)
        if cached is None:
            cached = self._dominators.reachable(node, forward=forward)
            self._reach[key] = cached
        return cached

    def any_between(
        self, source: CfgNode, target: CfgNode, candidates: Iterable[int],
    ) -> bool:
        """
        Whether any node in *candidates* — given as node ids — lies on some path from *source* to
        *target*.

        Asked as two walks rather than one path enumeration: a node lies between the two exactly when
        it is reachable forward from *source* and *target* is reachable forward from it, and the
        second half is the same as it being reachable backward from *target*. The forward set is
        intersected first and the backward walk is skipped entirely when that comes back empty, which
        is the common case for a caller whose candidate set is small.
        """
        candidates = frozenset(candidates)
        if not candidates:
            return False
        downstream = self.reachable(source, forward=True) & candidates
        if not downstream:
            return False
        return bool(downstream & self.reachable(target, forward=False))
