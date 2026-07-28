"""
Static-analysis substrate for PowerShell deobfuscation. Transforms query a shared, computed model of
the program here instead of each re-deriving scope, binding, effect, and liveness facts on their
own.

The foundation is `model`, a semantic model of scopes and resolved variable bindings, and
`callgraph`, which records what a command name denotes and who reaches it. On top of both, `effects`
decides what evaluating a node does, whether it can raise, and — through the call graph — where the
value a body writes to the output stream is finally read. Later layers (control-flow graphs,
interprocedural summaries) attach behind the same representation-agnostic surface.
"""
from __future__ import annotations

from refinery.lib.scripts.ps1.analysis.cache import Ps1ModelCache, model_cache
from refinery.lib.scripts.ps1.analysis.effects import (
    StatementEffect,
    is_side_effect_free,
    statement_effect,
)
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
    'StatementEffect',
    'build_semantic_model',
    'is_assignment_write_target',
    'is_side_effect_free',
    'is_write_occurrence',
    'model_cache',
    'replaces_value',
    'statement_effect',
]
