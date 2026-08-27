from __future__ import annotations

from refinery.lib.scripts.ps1.deobfuscation import deobfuscate
from refinery.lib.scripts.ps1.model import Ps1Script
from refinery.lib.scripts.ps1.parser import Ps1Parser
from refinery.lib.scripts.ps1.synth import Ps1Synthesizer
from refinery.lib.types import Param
from refinery.units import Arg
from refinery.units.scripting import IterativeDeobfuscator


class ps1(IterativeDeobfuscator):
    """
    AST-based PowerShell deobfuscator.

    Parses the script into an abstract syntax tree, applies simplifying transformations (constant
    folding, format string evaluation, bracket removal, type cast simplification, string
    operations, case normalization, invoke simplification, uncurly variables), and synthesizes
    clean output. Iterates until stable; running this twice does not change the output.

    **What the output preserves.** Every side effect the script performs, and every value that
    anything other than the console could read. In PowerShell a statement that merely yields a value
    has written to the success output stream, so a bare `'literal'` or `42` is output like any
    other. Such a statement is deleted only where three things are provable at once: that evaluating
    it cannot raise, that its value reaches the process output and nothing else — traced through
    every call site, so a bare value inside a function is kept unless every caller merely prints it
    — and that no redirection moves that output elsewhere. Anything that fails one of them is kept,
    as is every statement that does something, whatever it writes.

    That leaves a real class of junk standing. `[Math]::Sqrt(36)` and `Get-Random` are removed by
    neither model, because nothing here can prove a call does not throw.

    **Junk below an `Invoke-Expression`.** An `iex`, a `& $x`, a dot-sourced file or a `Set-Alias`
    to a computed target runs code this analysis cannot read, and such code can rename a .NET type,
    re-point a property or define a function over a cmdlet's name. Everything written below one is
    therefore kept, which on a script that decodes its payload halfway through can be the bulk of
    the output. The `-e` switch assumes such code changes none of that and removes the junk anyway;
    it is unsound by construction and meant for reading a script rather than running the result.

    **The assumption behind the default.** Stripping console output treats the input as a standalone
    script. A file cannot say whether it is a module: a `.psm1` exports its functions to callers no
    walk over this tree can see, and a bare value inside such a function is part of what those
    callers receive. Use the switch for a module, for a fragment that runs as part of something
    larger, or whenever the printed output is itself the artifact.
    """

    def __init__(
        self,
        timeout=500,
        keep_output: Param[bool, Arg.Switch('-k', help=(
            'Keep every statement that writes a value to the success output stream, including bare '
            'literals an obfuscator injected as noise. Use this when the input is a module or a '
            'fragment of a larger script, where such a value can reach a caller rather than only '
            'the console.'))] = False,
        trust_eval: Param[bool, Arg.Switch('-e', help=(
            'Assume that code the analysis cannot read - an Invoke-Expression, a call through a '
            'variable, a dot-sourced file, an alias bound to a computed target - leaves the .NET '
            'type system and the command table alone, and remove the junk written below it. This '
            'is unsound: such code can rename a type or shadow a cmdlet, so the output may behave '
            'differently. Use it to read a script, not to run the result.'))] = False,
    ):
        super().__init__(timeout=timeout, keep_output=keep_output, trust_eval=trust_eval)

    def parse(self, data: str) -> Ps1Script:
        return Ps1Parser(data).parse()

    def transform(self, ast: Ps1Script) -> int:
        return deobfuscate(
            ast,
            preserve_bare_output=self.args.keep_output,
            trust_eval=self.args.trust_eval,
        )

    def synthesize(self, ast: Ps1Script) -> str:
        return Ps1Synthesizer().convert(ast)
