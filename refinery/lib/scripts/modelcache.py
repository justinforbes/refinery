"""
Shared machinery for the per-run analysis model caches. A language builds one cache over the script
being transformed and shares it across every transform in a run, rebuilding its models only after
that script's tree changes — whether a transform announces the change through
`refinery.lib.scripts.Transformer.changed` or an in-pass mutation advances the script's
`refinery.lib.scripts.tree_version` counter — instead of each transform rebuilding from scratch on
every pass.

`refinery.lib.scripts.js.analysis.cache.ModelCache` and
`refinery.lib.scripts.ps1.analysis.cache.Ps1ModelCache` are the concrete caches: each subclass
declares its typed model slots in `_SLOTS` and exposes one lazy property per model built through
`_lazy`, and inherits the version tracking, edge-triggered invalidation, and transformer-reuse stash
from here. Keeping the mechanism in one place is why a new language's cache cannot drift from the
invalidation contract the base establishes.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from refinery.lib.scripts import Node, Transformer, tree_root, tree_version

_T = TypeVar('_T')
_C = TypeVar('_C', bound='ModelCacheBase')


class ModelCacheBase:
    """
    The version-tracking, invalidation, and reuse mechanism shared by every language's model cache.
    A subclass lists its lazily-built model attributes in `_SLOTS`, reads each through `_lazy`, and
    re-declares `root` at the node type it builds its models from. The base nulls the slots on
    construction, drops them together whenever this root's AST-mutation counter
    (`refinery.lib.scripts.tree_version`) advances past the value they were built at, and rebuilds
    on next access. Dropping the models together keeps a derived model consistent with the base
    model it was layered on. Because the base owns the whole mechanism, `invalidate` — the one
    method the `refinery.lib.scripts.AnalysisCache` protocol requires — is defined once, not per
    language.

    The run's caller-supplied `options` are held here rather than per language, because a model
    built differently under them is built once per cache and `for_transformer` is the one place a
    cache is created without a caller naming them. Taking them off the transformer there is what
    keeps the two from disagreeing: the pipeline sets the same object on both, and a cache that
    defaulted its own would answer under a configuration the transformer beside it does not hold.
    """

    _SLOTS: tuple[str, ...] = ()

    root: Node
    options: object | None

    def __init__(self, root: Node, options: object | None = None):
        # Normalized here and not only in `for_transformer`, because the version counter a mutation
        # advances is the one keyed on the tree: a cache holding a nested node as its root would
        # read a counter nothing ever bumps and never invalidate.
        root = tree_root(root)
        self.root = root
        self.options = options
        self._version = tree_version(root)
        self.invalidate()

    def invalidate(self) -> None:
        for slot in self._SLOTS:
            setattr(self, slot, None)

    def _ensure_fresh(self) -> None:
        version = tree_version(self.root)
        if version != self._version:
            self._version = version
            self.invalidate()

    def _lazy(self, slot: str, build: Callable[[], _T]) -> _T:
        """
        The value memoized in *slot*, built through *build* on first access after construction or
        an invalidation. Every model property routes through here so freshness is checked and the
        slot is filled by the one accessor primitive rather than a hand-copied check-build-store
        per model.
        """
        self._ensure_fresh()
        value = getattr(self, slot)
        if value is None:
            value = build()
            setattr(self, slot, value)
        return value

    @classmethod
    def for_transformer(cls: type[_C], transformer: Transformer, root: Node) -> _C:
        """
        The pipeline's shared cache for *root* when one of this exact class is attached to
        *transformer* and built over that same root, otherwise a fresh cache — stashed back onto
        *transformer* so later lookups within its single-pass lifetime reuse it instead of
        rebuilding the models per call. A transform still runs standalone (in tests, or outside the
        pipeline); freshness stays governed by the tree version, and a standalone mutation
        invalidates the stashed cache exactly as it would the shared one.

        *root* is normalized to the tree it belongs to — by the constructor, so the two entry points
        cannot disagree — and a transform that visits a nested body therefore means the same cache
        as one that visits the script. Skipping that leaves a whole-script model derived from a
        subtree: a leak sitting outside it becomes invisible and the world reads closed, which is
        the one direction that deletes code.

        A fresh cache takes the options the transformer is holding, which is the run's own
        configuration wherever the pipeline set it. Defaulting them here instead would let one
        transform in a run read a model built under a configuration the run never asked for, and
        stash that cache back for every transform after it.
        """
        root = tree_root(root)
        cache = transformer.models
        if isinstance(cache, cls) and cache.root is root:
            return cache
        cache = cls(root, transformer.options)
        transformer.models = cache
        return cache
