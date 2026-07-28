from __future__ import annotations

from refinery.lib.scripts.ps1.deobfuscation import deobfuscate
from refinery.lib.scripts.ps1.model import Ps1Script
from refinery.lib.scripts.ps1.parser import Ps1Parser
from refinery.lib.scripts.ps1.synth import Ps1Synthesizer
from refinery.units.scripting import IterativeDeobfuscator


class ps1(IterativeDeobfuscator):
    """
    AST-based PowerShell deobfuscator.

    Parses the script into an abstract syntax tree, applies simplifying transformations (constant
    folding, format string evaluation, bracket removal, type cast simplification, string
    operations, case normalization, invoke simplification, uncurly variables), and synthesizes
    clean output. Iterates until stable; running this twice does not change the output.

    **What the output preserves.** Every side effect the script performs, and every value it writes
    to an output stream. In PowerShell a statement that merely yields a value has written to that
    stream, so a bare `'literal'`, `42` or `[Math]::Sqrt(36)` is output like any other and survives
    — at the script root, where the host prints it, and inside a function, where it becomes part of
    what the caller receives. Statements are removed only when they provably write nothing and do
    nothing: an empty statement, the `$Null = ...` and `[Void]...` discard idioms, a pipeline ending
    in `Out-Null`, and code no path reaches.

    This costs recall on injected noise, deliberately. A junk literal an obfuscator padded the
    script with is indistinguishable, by any property of the *program*, from a literal the script
    exists to emit, and telling them apart would mean judging what the bytes are rather than what
    the code does. Deobfuscation here is a semantics-preserving rewrite; it is not a filter for
    what is worth reading.
    """

    def parse(self, data: str) -> Ps1Script:
        return Ps1Parser(data).parse()

    transform = staticmethod(deobfuscate)

    def synthesize(self, ast: Ps1Script) -> str:
        return Ps1Synthesizer().convert(ast)
