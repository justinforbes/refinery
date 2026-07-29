"""
The language-agnostic half of the script analysis substrate: the graph representation every
language's control-flow model is built out of, and the solvers that read it.

A language contributes two things and nothing else — the shape of its statements, expressed by
driving `refinery.lib.scripts.analysis.cfg.CfgBuilder`, and the oracles the solvers ask about
bindings and effects. Everything between those two is the same for every language, which is why it
lives here: `refinery.lib.scripts.modelcache.ModelCacheBase` established the pattern for the caches,
and this package is the same move for the flow layer.
"""
