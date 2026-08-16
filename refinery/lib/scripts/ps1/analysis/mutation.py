"""
What a call leaves in the slot it writes through.

`refinery.lib.scripts.ps1.analysis.arguments` says *which* slots a call writes; this says what is in
one afterwards. The two are separate because they are used separately: a caller refusing to install
a value only needs the first, and every refusal it makes is safe whether or not the second answers.

Nothing here computes a value. `[Array]::Reverse` reverses a collection, and what a collection is
and how one is spelled is `refinery.lib.scripts.ps1.analysis.values`' to say, so this reads the
value in and writes the value out through that module and holds only the rule between them.

**A rule is added only where the call is total over the values it accepts.** `[Array]::Sort` throws
on an array of mixed types, `[Array]::Clear` throws when the range runs off the end, and a rule that
answered for those would replace an error record with a value — worse than not answering, because a
refusal costs a fold and this would print something the script never printed. `Reverse` over a range
it fits in throws for no input, which is why it is the one rule here.
"""
from __future__ import annotations

from typing import Sequence

from refinery.lib.scripts import Expression, _clone_node
from refinery.lib.scripts.ps1 import data
from refinery.lib.scripts.ps1.analysis.model import written_call_slot
from refinery.lib.scripts.ps1.analysis.values import integer_of, read, unwrap_to_array_literal
from refinery.lib.scripts.ps1.model import (
    Ps1AccessKind,
    Ps1ArrayLiteral,
    Ps1TypeExpression,
    Ps1Variable,
)

#: The one member whose effect on its slot is written down here, as a canonical type key and a
#: lowercased member name.
_REVERSE = (data.required_type_key('array'), 'reverse')


def value_after(occurrence: Ps1Variable, previous: Expression) -> Expression | None:
    """
    The value the name *occurrence* stands for holds once the call around it has run, given the
    value `previous` it held before it, or `None` where no rule names one.

    `None` is the answer to every kind of doubt: a member with no rule, an overload the arity
    does not settle, a call that writes more than the one slot, a conversion standing between
    the name and the slot, an operand this cannot read as a collection, and a range that would
    throw. A caller must fold nothing on it.
    """
    found = written_call_slot(occurrence)
    if found is None or not found.written.settled or found.written.slots != {found.slot}:
        return None
    if found.slot < 0:
        # The receiver is a slot too, and it is not among the arguments: the bounds below are the
        # arguments *after* the written one, which for a receiver would be the whole list. No rule
        # here is about a receiver yet, and reading one that way would take the first argument of
        # `$x.SetValue(9, 0)` for a range.
        return None
    if found.through_a_part:
        # `[Array]::Reverse($p[0])` turns around the array `$p`'s first element is, so what `$p`
        # holds afterwards is the outer array with that one element replaced. Naming it is Knobe
        # and Sarkar's element update, which needs the index to be constant and the element to be
        # unshared; until it is built, answering the *slot* here would report the inner array's new
        # order as the outer array's.
        return None
    if found.through_a_conversion:
        # A cast between the name and the slot may hand the callee a fresh value built from what the
        # name holds rather than the value itself — measured, `[Array]::Reverse([int[]]$x)` over an
        # `Object[]` reverses a temporary and leaves `$x` in its original order. Which of the two a
        # cast is depends on the operand's runtime type, so the pair is refused rather than read.
        return None
    call = found.call
    if call.access is not Ps1AccessKind.STATIC or not isinstance(call.member, str):
        return None
    named = call.object
    if not isinstance(named, Ps1TypeExpression):
        return None
    resolved = data.resolve_type(named.name)
    if resolved is None or (resolved.generic_definition, call.member.lower()) != _REVERSE:
        return None
    return _reversed(previous, call.arguments[found.slot + 1:])


def _reversed(previous: Expression, bounds: Sequence[Expression]) -> Expression | None:
    """
    The collection `previous` names with a run of it turned around, or `None` where this names none.

    `bounds` is what the call passes after the array: nothing for the whole of it, an index and a
    length for a part. A part that does not fit inside the collection is refused rather than
    clamped, because 5.1 does not reverse anything there at all — measured,
    `[Array]::Reverse($b, 0, 99)` over three elements throws `ArgumentException`, and answering with
    a value would put one where the script raised.

    Only a collection is answered for. A String bound to the `System.Array` parameter is converted
    to a one-element `object[]` holding the string, which the call turns around and discards, so the
    variable is left holding exactly what it held — measured, and a fact about the *conversion*
    rather than about reversal, which is why it is refused here rather than answered.

    The elements are copied before the answer is built out of them. A node adopts the children
    it is handed, so building the answer over the ones still standing in the tree would leave
    the array the script wrote with children naming a node that is nowhere in it — and every
    guard that asks what encloses a statement climbs out of the tree from there. This answer is
    a value, not a rewrite, and a value has to be free of the tree it was read from whether the
    caller installs it or not.
    """
    array = unwrap_to_array_literal(previous)
    if array is None:
        return None
    if bounds:
        if len(bounds) != 2:
            return None
        start = integer_of(read(bounds[0]))
        length = integer_of(read(bounds[1]))
        if start is None or length is None:
            return None
        if start < 0 or length < 0 or start + length > len(array.elements):
            return None
    else:
        start, length = 0, len(array.elements)
    stop = start + length
    elements = [_clone_node(element) for element in array.elements]
    return Ps1ArrayLiteral(
        elements=[*elements[:start], *elements[start:stop][::-1], *elements[stop:]])
