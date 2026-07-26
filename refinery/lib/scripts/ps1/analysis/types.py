"""
The expression-type oracle of the PowerShell analysis substrate: the .NET type a PowerShell
expression evaluates to, traced through member-access chains, or `None` when it cannot be
determined. It reads the collected type surface in `refinery.lib.scripts.ps1.data` and the syntactic
accessors in `refinery.lib.scripts.ps1.ast`, and nothing else, so it sits at the analysis level
where both the effect layer (`refinery.lib.scripts.ps1.analysis.effects`) and the deobfuscation
transforms can reach it without either importing the other.

`resolve_expression_type` is the free-function core: one expression, one type or `None`. `TypeOracle`
is the value object the effect layer threads to reason about member reads. It carries the optional
variable and pipeline typing a semantic model supplies and exposes two views over one engine —
`TypeOracle.candidate_types`, the set of types a value could have, and `TypeOracle.resolve`, that set
collapsed to a single type when it is unambiguous. A method return or a declared cmdlet output can be
one of several types, which is why the set is the primitive and the single type the derived view.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from refinery.lib.scripts.ps1.ast import (
    extract_first_positional_string,
    get_command_name,
    get_member_name,
    unwrap_parens,
)
from refinery.lib.scripts.ps1.data import (
    OBJ_COMMANDS,
    PROPERTY_TYPES,
    TYPE_ARG_COMMANDS,
    VARIABLE_TYPES,
    WMI_CLASS_NAMES,
    WMI_COMMANDS,
    _resolve_type_name,
    command_output_types,
    static_overloads,
)
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1AccessKind,
    Ps1ArrayExpression,
    Ps1ArrayLiteral,
    Ps1CastExpression,
    Ps1CommandInvocation,
    Ps1ExpressionStatement,
    Ps1HereString,
    Ps1IntegerLiteral,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1StringLiteral,
    Ps1TypeExpression,
    Ps1Variable,
)

if TYPE_CHECKING:
    from refinery.lib.scripts.ps1.analysis.world import Ps1TypeWorld


def resolve_expression_type(
    expr: Expression,
    variable_types: dict[str, str] | None = None,
) -> str | None:
    """
    Trace the .NET type of a PowerShell expression by walking member access chains. Returns the
    lowercase full .NET type name, or `None` if the type cannot be determined.
    """
    unwrapped = unwrap_parens(expr)
    if not isinstance(unwrapped, Expression):
        return None
    expr = unwrapped
    if isinstance(expr, (Ps1StringLiteral, Ps1HereString)):
        return 'system.string'
    if isinstance(expr, Ps1IntegerLiteral):
        return 'system.int32'
    if isinstance(expr, Ps1ArrayLiteral):
        return 'system.array'
    if isinstance(expr, Ps1ArrayExpression):
        if (
            len(expr.body) == 1
            and isinstance(expr.body[0], Ps1ExpressionStatement)
            and isinstance(expr.body[0].expression, Ps1ArrayLiteral)
        ):
            return 'system.array'
    if isinstance(expr, Ps1Variable):
        key = expr.name.lower()
        if variable_types and key in variable_types:
            return variable_types[key]
        return VARIABLE_TYPES.get(key)
    if isinstance(expr, Ps1TypeExpression):
        return _resolve_type_name(expr.name)
    if isinstance(expr, Ps1CastExpression):
        return _resolve_type_name(expr.type_name)
    if isinstance(expr, Ps1CommandInvocation):
        cmd_name = get_command_name(expr)
        if cmd_name is not None:
            cmd_lower = cmd_name.lower()
            if cmd_lower in OBJ_COMMANDS:
                type_str = extract_first_positional_string(expr)
                if type_str is not None:
                    return _resolve_type_name(type_str)
            elif cmd_lower in WMI_COMMANDS:
                class_str = extract_first_positional_string(expr)
                if class_str is not None:
                    wmi_lower = class_str.lower()
                    if wmi_lower in WMI_CLASS_NAMES:
                        return wmi_lower
    if isinstance(expr, Ps1MemberAccess):
        if expr.object is None:
            return None
        obj_type = resolve_expression_type(expr.object, variable_types)
        if obj_type is None:
            return None
        member_name = get_member_name(expr.member)
        if member_name is None:
            return None
        return PROPERTY_TYPES.get((obj_type, member_name.lower()))
    return None


#: Commands whose declared `[OutputType]` is a trustworthy *superset* of what they emit at runtime,
#: not merely a lower bound. Most commands under-declare: one that forwards its input emits the
#: input's type, which it never lists — `Get-Random -InputObject $procs` returns a `Process`,
#: `Get-Content` on a non-filesystem provider returns whatever that provider yields — so trusting
#: the declaration lets the member gate prove `(...).Path` pure over an incomplete candidate set and
#: delete a live effect. Only commands that emit their own output and cannot pass input through
#: belong here; a read on any other command's result stays unresolved, and therefore kept.
_CLOSED_OUTPUT_CMDLETS = frozenset({
    'get-date',
})


class TypeOracle:
    """
    The type view the effect layer consults to decide whether reading a member has a side effect. It
    resolves a PowerShell expression to the .NET types its value could have, over one engine in two
    views: `candidate_types` yields every possibility, and `resolve` collapses them to a single type
    when that is unambiguous.

    The oracle carries whatever variable and pipeline typing a semantic model has derived; with none
    it still resolves everything the static surface alone determines: literals, casts, `New-Object`,
    WMI queries, the return of a static method call and a cmdlet's declared outputs. A populated
    oracle only adds the reads whose object is a typed variable or a pipeline item.

    It also carries the two command-table facts the gate consults: `world_closed_at`, whether the
    script's type system is unmutated (a present-member purity grant requires it), and
    `may_trust_command_name`, whether the metadata still describes what a command name runs (a grant
    must decline it when it does not). Both read the same world and both withhold when there is
    none, so a caller can never obtain a grant from one that the other would refuse.
    """

    def __init__(
        self,
        variable_types: dict[str, str] | None = None,
        world: Ps1TypeWorld | None = None,
    ):
        self._variable_types = variable_types
        self._world = world

    def with_variable_types(self, variable_types: dict[str, str] | None) -> TypeOracle:
        """
        This oracle with `variable_types` substituted, carrying the same world. The world is a
        whole-script fact and the typing is not, so a caller holding typing derived for one node
        layers it here rather than building a second oracle — which would be free to answer the
        world questions differently, and is how one script comes to have two disagreeing verdicts.
        """
        return TypeOracle(variable_types, self._world)

    def world_closed_at(self, node) -> bool:
        """
        Whether the .NET type world is closed at `node`: no code the script runs can have shadowed a
        member through the Extended Type System or remapped a type accelerator, so a present-member
        purity grant can be trusted. This delegates to the
        `refinery.lib.scripts.ps1.analysis.world.Ps1TypeWorld` a model has supplied; an oracle
        built without one answers `False`, so the member gate keeps every access.
        """
        return self._world is not None and self._world.world_closed_at(node)

    def may_trust_command_name(self, name: str, node) -> bool:
        """
        Whether the collected metadata still describes what the command `name` runs at `node`, so a
        site may act on the name — for typing or for purity. Two things stop it describing it, and
        both are the world's to answer: the script redefines the name where the walk can classify
        the redefinition, or the world is open at `node`, in which case a dot-sourced file, an
        imported module, an `iex`, an item cmdlet writing the `function:` provider or an opaque
        dispatch can bind *any* name to code this tree does not contain. Reading the shadow set
        alone would trust every name in exactly the scripts able to rebind them, and that set holds
        only the two spellings the classifier sees. An oracle built without a world trusts no name,
        matching `world_closed_at`: both questions withhold when there is nothing to ask.

        Named for the question a caller actually has rather than for the shadow set, because the
        answer is wider than the set: a reader who takes this for "is it redefined?" and narrows it
        back to that would reopen a hole that deletes code, and no `_grant` sits in the path to
        catch it.
        """
        if self._world is None:
            return False
        return self._world.world_closed_at(node) and not self._world.command_shadowed(name)

    def resolve(self, expr: Expression) -> str | None:
        """
        The single lowercase .NET type name the expression evaluates to when that is unambiguous,
        else `None`: `candidate_types` collapsed, so a lone candidate is the answer and zero or
        several are `None`. This is a strict superset of `resolve_expression_type` — it resolves every
        expression that one does and, additionally, a static method call or cmdlet whose result is a
        single type — so putting it into a transform in place of the free function would widen what
        resolves, which is deliberately not done where existing output depends on the narrower answer.
        """
        candidates = self.candidate_types(expr)
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def candidate_types(self, expr: Expression) -> frozenset[str]:
        """
        The set of lowercase full .NET type names the expression's value could have, each a key the
        type metadata resolves, or the empty set when the type cannot be determined. A static method
        call contributes the return its overloads agree on, and a cmdlet call the output types it
        declares; either can be several, so a caller reasoning about the value must have its
        conclusion hold for every candidate. The single-type forms — literals, variables, casts,
        `New-Object`, WMI, and property chains — are delegated to `resolve_expression_type` rather
        than re-derived here.
        """
        unwrapped = unwrap_parens(expr)
        if not isinstance(unwrapped, Expression):
            return frozenset()
        expr = unwrapped
        if isinstance(expr, Ps1InvokeMember):
            return self._static_method_candidates(expr)
        if isinstance(expr, Ps1CommandInvocation):
            return self._command_candidates(expr)
        single = resolve_expression_type(expr, self._variable_types)
        return frozenset() if single is None else frozenset({single})

    def _static_method_candidates(self, node: Ps1InvokeMember) -> frozenset[str]:
        """
        The return type of a `[Type]::Method(...)` call, taken only when every matching static
        overload agrees on it; disagreement or an unrecognized call is the empty set. An instance
        method call is not resolved — its receiver type would have to be traced and its overloads
        selected by argument type — so it contributes nothing rather than a guess.
        """
        if node.access is not Ps1AccessKind.STATIC:
            return frozenset()
        obj = node.object
        member = node.member
        if not isinstance(obj, Ps1TypeExpression) or not isinstance(member, str):
            return frozenset()
        returns = {
            overload['returns'].lower()
            for overload in static_overloads(obj.name, member)
            if overload.get('returns')
        }
        return frozenset(returns) if len(returns) == 1 else frozenset()

    def _command_candidates(self, cmd: Ps1CommandInvocation) -> frozenset[str]:
        """
        The types a command's result could have: the constructed or queried type for the `New-Object`
        and WMI forms the single-type ladder already knows, otherwise the output types a command
        declares through `[OutputType]` — but only for a command whose declaration is a trustworthy
        *superset* of what it emits (`_CLOSED_OUTPUT_CMDLETS`). `[OutputType]` is a lower bound in
        general: a command that forwards its input emits types it never declares, and trusting the
        declaration there would let the member gate prove an effectful read pure over an incomplete
        candidate set. Every other command contributes nothing, so a read on its result stays
        unresolved and is kept.
        """
        name = get_command_name(cmd)
        if name is None:
            return frozenset()
        lower = name.lower()
        if not self.may_trust_command_name(lower, cmd):
            return frozenset()
        if lower in TYPE_ARG_COMMANDS:
            single = resolve_expression_type(cmd, self._variable_types)
            return frozenset() if single is None else frozenset({single})
        if lower not in _CLOSED_OUTPUT_CMDLETS:
            return frozenset()
        declared = command_output_types(name)
        return declared if declared is not None else frozenset()
