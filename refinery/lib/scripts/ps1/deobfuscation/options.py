"""
Caller-supplied options controlling PowerShell deobfuscation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ps1DeobfuscationOptions:
    """
    Options that steer PowerShell deobfuscation. *preserve_bare_output* selects what a statement
    that only writes a value to the success output stream is worth.

    - Stripping model (default, `preserve_bare_output=False`): such a statement is deleted wherever
      the analysis can prove three things: that evaluating it cannot raise, that its value reaches
      the process output and nothing more, and that no redirection moves it away. What is lost is
      text on a console nobody is watching, and what is bought is the removal of the junk an
      obfuscator pads a script with. The proof rests on one assumption no file can settle: that the
      input is a standalone script and not a library some other file imports, whose functions are
      then called from call sites this analysis never sees.

    - Preserving model (`preserve_bare_output=True`): no such statement is ever deleted. This is the
      answer for a script whose printed output *is* the artifact, and for a `.psm1` or any other
      fragment that runs as part of something larger.

    Neither model touches a statement whose value is captured, whose evaluation does anything, or
    that this analysis cannot read; those are kept under both, and the switch is not what protects
    them.
    """
    preserve_bare_output: bool = False


def bare_output_is_preserved(options: object | None) -> bool:
    """
    Whether *options* asks for every write to the success output stream to be kept. Any value that
    is not a `Ps1DeobfuscationOptions` — a transformer run standalone, or one with no options
    attached — defaults to the stripping model, which is what the pipeline does unless told.
    """
    return isinstance(options, Ps1DeobfuscationOptions) and options.preserve_bare_output
