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

The two owners divide by what the pass is claiming, and the division is what makes the rule below
sound. A substitution claims the value is unchanged, so everything the original did still has to
happen. A removal claims the code does not run at all — `Ps1DeadCodeElimination` resolving a
constant `if` deletes the branch that was never taken, redirections and all — and that claim is
`refinery.lib.scripts.ps1.deobfuscation.removal.Ps1RemovalPlan`'s to check. One rule over both would
refuse the second for no reason.
"""
from __future__ import annotations

from typing import Iterable

from refinery.lib.scripts import (
    Node,
    Statement,
    _replace_in_parent,
    reattach,
    set_child,
    set_child_list,
)
from refinery.lib.scripts.ps1.deobfuscation.removal import Ps1RemovalPlan

#: What every entry point below accepts on either side of a substitution: a node, nothing, or a
#: list of either. A pass that splits one statement into several has a list on one side and a single
#: node on the other, and nothing about the rule changes between the two.
Part = Node | None | Iterable['Part']


def _nodes(part: Part) -> Iterable[Node]:
    for item in part if isinstance(part, (list, tuple)) else [part]:
        if isinstance(item, Node):
            yield item


def carried_redirections(*parts: Part) -> list:
    """
    Every redirection written anywhere in the subtrees at `parts`.

    The whole subtree is read and not the nodes themselves, because the carrier is rarely the node a
    pass is holding: `iex 'x' > C:\\o.txt` writes the redirection on the invocation, and a pass
    replacing the expression statement around it is handed a node whose own list is empty. The list
    is read by name rather than matched against the model types known to carry one, so a carrier
    added later is covered without being enumerated here — and the failure of the alternative is
    asymmetric, since a carrier this did not recognize reads as nothing being lost.
    """
    found = []
    for part in parts:
        for node in _nodes(part):
            for descendant in node.walk():
                found.extend(getattr(descendant, 'redirections', None) or ())
    return found


def may_substitute(removed: Part, installed: Part, moved: Part = ()) -> bool:
    """
    Whether `installed` may take the place of `removed` without losing what the original wrote.

    A replacement expression carries no redirections, so folding `f > C:\\log` into the value `f`
    returns puts that value on the console and leaves the file uncreated — PowerShell opens the
    target as it sets the redirection up, so the file is touched even by a command that writes
    nothing at all. **Every spelling is refused, including the merges that neither move output nor
    open a file.** `f 2>&1` folds error records into the output stream, so dropping it is
    observationally identical only when `f` writes nothing to stderr, and nothing here can know
    that. What this asks is whether the replacement can carry the redirection, and it never can.

    `moved` names nodes the caller lifts out of `removed` and re-installs elsewhere within the same
    edit. `Ps1ExpandableStringHoist` rewrites `"a$($z = f > C:\\o.txt)b"` by replacing the string
    with its text and hoisting the assignment into the enclosing body, so the redirection leaves
    this substitution and returns one statement later; without saying so the pass would be refused
    for an edit that loses nothing. Nothing checks the claim — a caller that names a node here and
    then drops it loses exactly what this exists to keep — so it is owed by the caller, the way
    `refinery.lib.scripts.ps1.deobfuscation.removal.Ps1RemovalPlans.propose_in` owes the list it
    names.

    Identity decides whether a redirection survived, not equality. A pass that rebuilt one would be
    refused although it wrote the same thing down, which is the conservative direction: what a
    wrong answer the other way costs is a payload deleted into a file that is never created.
    """
    kept = {id(redirection) for redirection in carried_redirections(installed, moved)}
    return all(id(redirection) in kept for redirection in carried_redirections(removed))


def _release(part: Part) -> bool:
    """
    Put everything `part` holds back in order and report the refusal.

    Building a replacement adopts the parts of the original it reuses, so a rejected one leaves
    those parts standing in the tree while naming a holder that does not hold them, and everything
    that reads upward from inside one then walks out of the tree.
    `refinery.lib.scripts.ps1.deobfuscation.removal.Ps1RemovalPlan.propose` makes the same repair
    for the same reason.
    """
    for node in _nodes(part):
        reattach(node)
    return False


def substitute(old: Node, new: Node, moved: Part = ()) -> bool:
    """
    Put `new` where `old` stands, reporting whether it landed. Refused when the swap would lose a
    redirection; see `may_substitute`, whose `moved` argument this forwards.

    `old` may sit in a direct field, in a child list, or in a tuple inside one, and the search is
    the same one a removal makes, so a pass that found its node by walking the tree does not have to
    know which of the three it is looking at.
    """
    if not may_substitute(old, new, moved):
        return _release(old)
    return _replace_in_parent(old, new)


def substitute_field(parent: Node, attr: str, new: Node | None) -> bool:
    """
    Put `new` in the single-node field `parent.<attr>`, reporting whether it landed.

    A `None` clears the field, which is how a pass drops an optional sub-node: a loop's condition
    that has been proved constant, or the expression of a statement whose work has been folded into
    a neighbour. Clearing is where the rule earns its keep here — what leaves is everything the
    field held and what arrives is nothing at all.
    """
    old = getattr(parent, attr, None)
    if not may_substitute(old, new):
        return _release(old)
    set_child(parent, attr, new)
    return True


def substitute_list(parent: Node, attr: str, items: list) -> bool:
    """
    Replace the contents of the child list `parent.<attr>` with `items`, reporting whether they
    landed.

    The new contents are given whole rather than as a diff, so a reordering and a wholesale rewrite
    arrive here in the same shape, and what either of them takes out of the tree is read off the two
    lists rather than taken from the caller's account of it.
    """
    old = list(getattr(parent, attr, None) or [])
    if not may_substitute(old, items):
        return _release(old)
    set_child_list(parent, attr, items)
    return True


def substitute_statement(
    container: Node,
    statement: Statement,
    replacement: list[Statement],
    moved: Part = (),
) -> bool:
    """
    Put `replacement` where `statement` stands in `container.body`, reporting whether the body
    moved.

    `replacement` has to hold something. A pass that wants the statement gone opens a
    `refinery.lib.scripts.ps1.deobfuscation.removal.Ps1RemovalPlan` itself, because the guards that
    decide whether a statement may go belong to that class, and every one of them would be skipped
    by a caller that spelled its removal as a substitution with nothing on the other side.

    The rule is asked of the statement against the whole replacement list and not against any one
    entry of it, because a pass that splits one statement into several routinely puts the redirected
    part in a different entry than the value.
    """
    if not replacement:
        raise ValueError('an empty replacement is a removal; open a Ps1RemovalPlan for it')
    if not may_substitute(statement, replacement, moved):
        return _release(statement)
    plan = Ps1RemovalPlan(container)
    plan.propose(statement, replacement)
    return plan.commit()
