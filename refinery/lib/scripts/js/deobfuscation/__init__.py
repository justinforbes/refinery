"""
JavaScript AST deobfuscation transforms.
"""
from __future__ import annotations

from refinery.lib.scripts.js.deobfuscation.simplify import JsSimplifications
from refinery.lib.scripts.js.deobfuscation.stringarray import JsStringArrayResolver
from refinery.lib.scripts.js.deobfuscation.wrappers import JsCallWrapperInliner
from refinery.lib.scripts.js.model import JsScript
from refinery.lib.scripts.pipeline import DeobfuscationPipeline, TransformerGroup

_pipeline = DeobfuscationPipeline(
    groups=[
        TransformerGroup('wrappers', JsCallWrapperInliner),
        TransformerGroup('stringarray', JsStringArrayResolver),
        TransformerGroup('simplify', JsSimplifications),
    ],
    dependencies={
        'stringarray': {'wrappers'},
        'simplify': {'stringarray'},
    },
)


def deobfuscate(ast: JsScript, max_steps: int = 0) -> int:
    """
    Apply all available deobfuscators to the input.
    """
    return _pipeline.run(ast, max_steps=max_steps)
