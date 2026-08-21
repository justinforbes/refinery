"""
The instruments the two JavaScript defect ledgers are written with.

`test.lib.scripts.js.test_release_blockers` holds the defects a release is held for and
`test.lib.scripts.js.test_unfixed_defects` holds the rest, so an entry moves from one file to the
other when what it costs is reassessed rather than when it is understood differently. Both are
written against the same few questions, which live here so that neither file can answer one of them
its own way and read as disagreeing with the other about a defect they both pin.

Nothing here executes anything from the sample corpus: `before_and_after` runs Node, and every
program it is ever given is one an entry wrote out by hand.
"""
from __future__ import annotations

from test.lib.scripts.js.analysis.differential import behavior, deobfuscate_source

from refinery.lib.scripts import is_well_formed
from refinery.lib.scripts.js.parser import JsParser
from refinery.units.scripting.js import js


def well_formed(source: str) -> bool:
    return is_well_formed(JsParser(source).parse())


def folded(source: str) -> str:
    return source.encode('utf8') | js() | str


def before_and_after(source: str) -> tuple[tuple[str, str | None], tuple[str, str | None]]:
    """
    What Node makes of *source* and what it makes of the text `refinery.js` deobfuscates it to,
    reported together because the law is that the two agree.
    """
    return behavior(source), behavior(deobfuscate_source(source))


def each_program_still_prints(
    programs: dict[str, str],
) -> dict[str, tuple[tuple[str, str | None], tuple[str, str | None]]]:
    """
    The pair `before_and_after` has to give for each program in *programs*: the text the program
    prints, printed by the deobfuscation too, with neither of the two throwing.
    """
    return {
        source: ((prints, None), (prints, None))
        for source, prints in programs.items()
    }
