"""
The .NET type a variable carries where it is read.

A variable has no type of its own — the value it holds does — so the question is which write a read
observes and what type that write establishes. The first half is
`refinery.lib.scripts.ps1.analysis.dataflow.Ps1VariableFlow.reaching_definition` and the second is
`refinery.lib.scripts.ps1.analysis.values.resolve_expression_type`, so what is left here is the
join, exactly as `refinery.lib.scripts.ps1.analysis.separator` joins the same two layers for `$OFS`.

**A name is not a variable, and that is what this is for.** What stood before it scanned the whole
script for names written exactly once and offered the type to every read of that name anywhere,
which crosses every scope boundary the language has: a write inside a function body typed a read at
the top level, so `function f { $q = New-Object Net.WebClient }` followed by
`($q | Get-Member)[0].Name` folded to a member of a type the top-level `$q` never holds — it holds
`$null`, and `Get-Member` over `$null` is an error rather than a member list. Counting writes also
refused a name written twice however far apart, so a type that plainly reaches its read was thrown
away beside the one that does not.
"""
from __future__ import annotations

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.analysis.dataflow import Ps1FlowUnknown, Ps1VariableFlow
from refinery.lib.scripts.ps1.analysis.values import resolve_expression_type
from refinery.lib.scripts.ps1.data import resolve_type
from refinery.lib.scripts.ps1.dotnet import Ps1TypeName
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1ForEachLoop,
    Ps1HereString,
    Ps1StringLiteral,
    Ps1Variable,
)

#: The reasons a binding's values cannot be tracked that no reasoning about its *type* can survive:
#: each of the three is a way for something this layer does not see to store a value of any type
#: under the name, so nothing the source spells about the name is a claim about what it holds.
#: A write nothing can place and a store through a member are not among them — they say which write
#: ran cannot be settled, and where the writes agree that question has no bearing on the type;
#: `$x.Length = 5` leaves `$x` naming the object it already named. Writes spread over several
#: bodies are the third of that kind, and `type_at` refuses them on its own account rather than
#: here, because its first rule can still name one of them.
_INSTALLS_ANY_TYPE = (
    Ps1FlowUnknown.REACHED_BY_QUALIFIER
    | Ps1FlowUnknown.WRITTEN_BY_DEFERRED_BODY
    | Ps1FlowUnknown.WRITTEN_BY_UNREADABLE_NAME
)


def type_at(var: Ps1Variable, flow: Ps1VariableFlow) -> Ps1TypeName | None:
    """
    The type of the value *var* reads where it stands, or `None` where no single type is.

    **Two rules, and either one alone answers.** The write the read observes is the first and the
    precise one: it names the type even where the binding's writes disagree. Where the ordering
    names no single write, a binding *all* of whose writes establish the same type still has that
    type at every read of it, because which of them ran cannot matter. The second rule is what
    keeps this answering at the positions the ordering declines and a type survives anyway — a
    name stored through a member, and a write and a read inside one `$( ... )`, which the graphs
    hold as a single point — and dropping it took the member spellings off a real sample.

    The second rule has to establish two things the first gets for free. **That a write has run at
    all**, which `written_before` answers: writes that agree say nothing about a read that precedes
    every one of them, where the name holds what it held before the script started. And **that the
    writes it is agreeing over are all of them**, which `foreign_write_before` answers:
    `Invoke-Expression $code` stores a value of any type under any name and `. { $q = … }` stores
    into the caller's scope from a binding of its own, so agreement among the writes this binding
    holds is no claim about the ones it does not. Writes spread over several bodies are refused
    there too, because whether one body runs before another is what this layer does not answer.
    """
    binding = flow.semantic.binding_of(var)
    if binding is None or not binding.writes:
        return None
    if flow.unknowns(binding) & _INSTALLS_ANY_TYPE:
        return None
    observed = flow.reaching_definition(var)
    if observed is not None:
        return _established_by(observed)
    if flow.unknowns(binding) & Ps1FlowUnknown.WRITES_IN_SEVERAL_BODIES:
        return None
    if flow.foreign_write_before(var) or not flow.written_before(var):
        return None
    named = {_established_by(write.node) for write in binding.writes}
    return named.pop() if len(named) == 1 else None


def _established_by(write: Node) -> Ps1TypeName | None:
    """
    The type a write occurrence puts into the name, or `None` for a write whose type this cannot
    name.

    Only the two writes that carry a type with them are read. A plain assignment establishes the
    type of what was assigned, and a `foreach` binding the type of one element of what is iterated.
    Everything else — a compound assignment, a `[ref]` cast, a parameter with no default, a `++`,
    a name a command writes — is a write whose type is not written down beside it, and a write
    this cannot type is one it must refuse rather than look past: looking past it is what the
    whole-script scan did.

    The assigned expression is typed with no variable typing of its own, so `$b = $a` leaves `$b`
    untyped rather than chasing `$a`'s write in turn. That is what the scan did as well, and the
    chase is a recursion this would have to bound — `$x = $x` inside a loop is a cycle in it.
    """
    parent = write.parent
    if isinstance(parent, Ps1AssignmentExpression):
        if parent.target is not write or parent.operator != '=':
            return None
        value = parent.value
        return None if not isinstance(value, Expression) else resolve_expression_type(value)
    if isinstance(parent, Ps1ForEachLoop) and parent.variable is write:
        return _element_type(parent.iterable)
    return None


#: What `foreach` yields from a string: the string itself, not its characters.
_STRING = resolve_type('System.String')


def _element_type(iterable: Expression | None) -> Ps1TypeName | None:
    """
    The type of one element a `foreach` iterable yields, or `None` where its elements do not share
    one this can name. A string yields itself rather than its characters, which is measured.
    """
    if isinstance(iterable, (Ps1StringLiteral, Ps1HereString)):
        return _STRING
    if not isinstance(iterable, Ps1ArrayLiteral) or not iterable.elements:
        return None
    named = set()
    for element in iterable.elements:
        if not isinstance(element, Expression):
            return None
        one = resolve_expression_type(element)
        if one is None:
            return None
        named.add(one)
    return named.pop() if len(named) == 1 else None
