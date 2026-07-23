"""
Static-analysis substrate for PowerShell deobfuscation. Transforms query a shared, computed model of
the program here instead of each re-deriving scope, binding, and liveness facts on their own.

The foundation is `model`, a semantic model of scopes and resolved variable bindings. Later layers
(effect summaries, control-flow graphs) attach behind the same representation-agnostic surface.
"""
from __future__ import annotations

from refinery.lib.scripts.ps1.analysis.cache import Ps1ModelCache, model_cache
from refinery.lib.scripts.ps1.analysis.model import (
    Binding,
    Ps1SemanticModel,
    Scope,
    ScopeKind,
    build_semantic_model,
    is_assignment_write_target,
    is_write_occurrence,
    replaces_value,
)

__all__ = [
    'Binding',
    'Ps1ModelCache',
    'Ps1SemanticModel',
    'Scope',
    'ScopeKind',
    'build_semantic_model',
    'is_assignment_write_target',
    'is_write_occurrence',
    'model_cache',
    'replaces_value',
]
