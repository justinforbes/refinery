"""
What a collection is written with between its elements where PowerShell coerces one to a String.

The separator is `$OFS`, and it is an ordinary variable rather than anything the engine maintains:
the conversion looks the name up in the scope chain wherever it happens, and a script may write it
like any other. So the question a fold has to answer before writing a collection down is a dataflow
question — `refinery.lib.scripts.ps1.analysis.dataflow.Ps1VariableFlow.write_observed_at` is the
whole of it — and what is left here is the part about the *conversion* rather than about the name:
what an unwritten `$OFS` means, and which values may be read as a separator at all.

**The single space is the join's fallback, not a value the name holds.** Measured: `$OFS` does not
exist in a fresh session, `Test-Path variable:OFS` is `False` and `$null -eq $OFS` is `$true`, and
`[string]@(1, 2)` is still `1 2`. The difference is observable, which is why the two may never be
folded together: `$OFS = $null` separates with a space and `$OFS = ''` separates with nothing.

**Only an implicit conversion reads it.** Under `$OFS = ','`, measured: `[string]@(1, 2)` is `1,2`,
`"$(1, 2)"` is `1,2` and `'a' + @(1, 2)` is `a1,2`, where `@(1, 2) -join '-'`, `-join @(1, 2)` and
`[string]::Join('-', @(1, 2))` name their own separator and are unmoved by it. A caller that names a
separator must not ask here. The concatenation is the one implicit conversion that does not ask
either, because it is answered inside the value domain's own kernel, which holds no point to ask at.

**The separator is written by the current culture and the elements are not.** Measured on a host
whose culture writes a decimal comma: `$OFS = 1.5` separates with `1,5`, while `[string]@(1.5, 2)`
is `1.5 2`. So the elements are `coerced_text` and the separator is
`refinery.lib.scripts.ps1.analysis.values.invariant_text`, which refuses the values the two readings
disagree about.
"""
from __future__ import annotations

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.analysis.dataflow import Ps1ObservedWrite, Ps1VariableFlow
from refinery.lib.scripts.ps1.analysis.values import (
    NULL,
    collect_texts,
    invariant_text,
    read,
)
from refinery.lib.scripts.ps1.model import Ps1AssignmentExpression, Ps1Variable

#: The key `refinery.lib.scripts.ps1.analysis.model.binding_key` files `$OFS` under. A scope
#: qualifier does not change it, so `$script:OFS` is a write of this same name.
_OFS = 'ofs'

#: What a join writes between two elements where nothing has written `$OFS`, and what one written
#: `$null` writes as well — see this module's own documentation for why those are the same answer
#: and are not the answer for `''`.
_FALLBACK = ' '


def output_field_separator_at(site: Node, flow: Ps1VariableFlow) -> str | None:
    """
    The text a collection coerced to a String at *site* is written with between its elements, or
    `None` where no single text is.

    `None` is the answer to every kind of doubt and a caller must fold nothing on it. That includes
    the case an obfuscated script arrives in: a payload the analysis has not resolved yet may write
    `$OFS` before the conversion, and while it stands there this refuses. The iteration is what
    settles it — once the payload is literal, the write it performs is one this can see or one
    that is not there.
    """
    observed = flow.write_observed_at(_OFS, site)
    if observed is Ps1ObservedWrite.UNKNOWN:
        return None
    if observed is Ps1ObservedWrite.NOTHING:
        return _FALLBACK
    return _separator_written_by(observed)


def coerced_text_at(node: Node | None, site: Node, flow: Ps1VariableFlow) -> str | None:
    """
    The text an expression contributes where PowerShell coerces its value to a String at *site*, or
    `None` where this names none.

    `refinery.lib.scripts.ps1.analysis.values.coerced_text` is the same question about one value,
    and this is what a caller asks in its place wherever the value may be a *collection*: each
    element contributes what that one answers for it, and what stands between them is a fact about
    the point rather than about the value. A collection of fewer than two elements has nothing
    between them, so it is answered without asking after the separator at all — measured,
    `[string]@(1)` is `1` and `[string]@()` is the empty string whatever `$OFS` holds.
    """
    parts = collect_texts(node)
    if parts is None:
        return None
    if len(parts) < 2:
        return ''.join(parts)
    separator = output_field_separator_at(site, flow)
    return None if separator is None else separator.join(parts)


def _separator_written_by(write: Node) -> str | None:
    """
    The separator a write of `$OFS` establishes, or `None` for one whose value this cannot read.

    Only a plain assignment of a value straight to the name is read. A command that writes the name
    — `Set-Variable OFS '-'` — is a write this refuses rather than reads, and so is a compound
    assignment, a `foreach` binding and a multi-assignment; the write is still a write, and having
    been counted as one is what makes refusing it safe.

    A type constraint is not transparent here although
    `refinery.lib.scripts.ps1.ast.assignment_of` reads through one: `[int]$OFS = '0x10'` stores what
    the constraint converts, not what was written, so the target must be the name itself.
    """
    if not isinstance(write, Ps1Variable):
        return None
    assignment = write.parent
    if not isinstance(assignment, Ps1AssignmentExpression):
        return None
    if assignment.target is not write or assignment.operator != '=':
        return None
    fact = read(assignment.value)
    return _FALLBACK if fact is NULL else invariant_text(fact)
