"""
A per-run cache of the PowerShell analysis models. The deobfuscation pipeline builds one cache over
the script being transformed and shares it across every transform in a run, rebuilding the model
only after that script's tree changes — whether a transform announces the change through
`refinery.lib.scripts.Transformer.changed` or an in-pass mutation advances the script's
`refinery.lib.scripts.tree_version` counter. The version tracking, invalidation, and
transformer-reuse mechanism live in `refinery.lib.scripts.modelcache.ModelCacheBase`; this module
only declares the PowerShell model slot and its `build_*` wiring.
"""
from __future__ import annotations

from refinery.lib.scripts import Transformer
from refinery.lib.scripts.modelcache import ModelCacheBase
from refinery.lib.scripts.ps1.analysis.model import Ps1SemanticModel, build_semantic_model
from refinery.lib.scripts.ps1.analysis.world import Ps1TypeWorld, build_closed_world
from refinery.lib.scripts.ps1.model import Ps1Script


class Ps1ModelCache(ModelCacheBase):
    """
    Lazily builds and memoizes the analysis models for one root script — the
    `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel` and the
    `refinery.lib.scripts.ps1.analysis.world.Ps1TypeWorld` — each dropped whenever this root's
    AST-mutation counter advances past the value it was built at. Later phases add a `control_flow`
    slot behind the same shape.
    """

    _SLOTS = ('_model', '_closed_world')

    root: Ps1Script
    _model: Ps1SemanticModel | None
    _closed_world: Ps1TypeWorld | None

    @property
    def model(self) -> Ps1SemanticModel:
        return self._lazy('_model', lambda: build_semantic_model(self.root))

    @property
    def closed_world(self) -> Ps1TypeWorld:
        return self._lazy('_closed_world', lambda: build_closed_world(self.root))


def model_cache(transformer: Transformer, root: Ps1Script) -> Ps1ModelCache:
    """
    The pipeline's shared `Ps1ModelCache` for *root* when one is attached to *transformer* and
    built over that same root, otherwise a fresh cache stashed back onto *transformer* for reuse
    within its single-pass lifetime. See
    `refinery.lib.scripts.modelcache.ModelCacheBase.for_transformer`.
    """
    return Ps1ModelCache.for_transformer(transformer, root)
