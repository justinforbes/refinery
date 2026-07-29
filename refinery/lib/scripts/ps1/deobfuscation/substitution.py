"""
The single route by which a PowerShell cleanup pass puts one part of the tree in another's place.

`refinery.lib.scripts.ps1.deobfuscation.removal` owns the other half of the same question: what a
pass takes *out* of a statement list. Between them there was nothing, so a pass that rewrote a node
sitting in a slot reached for the tree primitives directly, and any rule about what such a rewrite
must not throw away had to be restated at every site that could break it. The sites disagree, which
is how a redirected call came to be folded into its value in three different passes and left the
file it named uncreated.

The entry points are named for the slot the node sits in — a direct field, one entry of a child
list, the whole of one — rather than for the passes that reach them, because the slot is the only
thing they differ in. `substitute_statement` is the exception in that it edits nothing itself: a
statement list is a body, a body is edited as a batch, and
`refinery.lib.scripts.ps1.deobfuscation.removal.Ps1RemovalPlan` is that batch. It is here so that a
pass substituting inside a body has an owner to ask rather than a `refinery.lib.scripts.BodyEdit` to
build, and it refuses an empty replacement: whether a statement may go is the other module's
question, and answering it here would be a route around every guard that module holds.
"""
from __future__ import annotations

from refinery.lib.scripts import (
    Node,
    Statement,
    _replace_in_parent,
    set_child,
    set_child_list,
)
from refinery.lib.scripts.ps1.deobfuscation.removal import Ps1RemovalPlan


def substitute(old: Node, new: Node) -> bool:
    """
    Put `new` where `old` stands, reporting whether a slot was found for it.

    `old` may sit in a direct field, in a child list, or in a tuple inside one, and the search is
    the same one a removal makes, so a pass that found its node by walking the tree does not have to
    know which of the three it is looking at.
    """
    return _replace_in_parent(old, new)


def substitute_field(parent: Node, attr: str, new: Node | None) -> bool:
    """
    Put `new` in the single-node field `parent.<attr>`, reporting whether it landed.

    A `None` clears the field, which is how a pass drops an optional sub-node: a loop's condition
    that has been proved constant, or the expression of a statement whose work has been folded into
    a neighbour.
    """
    set_child(parent, attr, new)
    return True


def substitute_list(parent: Node, attr: str, items: list) -> bool:
    """
    Replace the contents of the child list `parent.<attr>` with `items`, reporting whether they
    landed.

    The new contents are given whole rather than as a diff, so a reordering and a wholesale rewrite
    arrive here in the same shape and what each of them takes out of the tree is read off the two
    lists rather than taken from the caller's account of it.
    """
    set_child_list(parent, attr, items)
    return True


def substitute_statement(
    container: Node,
    statement: Statement,
    replacement: list[Statement],
) -> bool:
    """
    Put `replacement` where `statement` stands in `container.body`, reporting whether the body
    moved.

    `replacement` has to hold something. A pass that wants the statement gone opens a
    `refinery.lib.scripts.ps1.deobfuscation.removal.Ps1RemovalPlan` itself, because the guards that
    decide whether a statement may go belong to that class and every one of them would be skipped by
    a caller that spelled its removal as a substitution with nothing on the other side.
    """
    if not replacement:
        raise ValueError('an empty replacement is a removal; open a Ps1RemovalPlan for it')
    plan = Ps1RemovalPlan(container)
    plan.propose(statement, replacement)
    return plan.commit()
