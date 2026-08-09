"""
.NET type system utilities for PowerShell deobfuscation.
"""
from __future__ import annotations

from refinery.lib.scripts import Node, Transformer
from refinery.lib.scripts.ps1.analysis.values import resolve_expression_type
from refinery.lib.scripts.ps1.ast import get_command_name, get_member_name, unwrap_parens
from refinery.lib.scripts.ps1.data import (
    CANONICAL_TYPE_NAMES,
    GET_MEMBER_ALIASES,
    TYPE_ACCELERATORS,
    canonical_member,
    resolve_type,
    view_members,
)
from refinery.lib.scripts.ps1.data import resolve_member_type as data_member_type
from refinery.lib.scripts.ps1.dotnet import Ps1TypeName, parse_type_name
from refinery.lib.scripts.ps1.deobfuscation.helpers import (
    MutationKind,
    iter_variable_mutations,
    make_string_literal,
)
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1CommandInvocation,
    Ps1ForEachLoop,
    Ps1HereString,
    Ps1IndexExpression,
    Ps1IntegerLiteral,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1Pipeline,
    Ps1PipelineElement,
    Ps1ScopeModifier,
    Ps1StringLiteral,
)


def _accelerator_spelling(name: str) -> bool:
    """
    Whether `name` is written as a type accelerator, with or without array, pointer or reference
    suffixes. A name the type-name grammar does not understand is judged as written, which is what
    it was before the suffixes were understood at all.
    """
    parsed = parse_type_name(name)
    if parsed is None:
        return name.lower() in TYPE_ACCELERATORS
    return parsed.name.lower() in TYPE_ACCELERATORS


def canonical_type_name(name: str) -> str | None:
    """
    Return the canonical PascalCase display name for a .NET type, preserving an explicit `System.`
    prefix if the caller used one. Returns `None` if the type is not in the database, or if the name
    is already a type accelerator: `[ref]`, `[int]` and `[string]` are the readable forms and are
    left untouched, where a verbose `[System.Int32]` still folds to `[Int32]`.

    The array suffixes are carried through rather than looked up, because the display table is keyed
    by the element type: rendering `byte[]` off its definition alone answers `Byte`, which is a
    different type, and `New-Object Byte[] $n` became `New-Object Byte $n`.

    An accelerator keeps its spelling through a suffix too. `[byte[]]` is the readable form for the
    same reason `[byte]` is, and it reached the display table only because the name it was looked up
    under carried the suffix and the table is keyed without one.
    """
    if _accelerator_spelling(name):
        return None
    resolved = resolve_type(name)
    lower = resolved.definition.lower() if resolved is not None else name.lower()
    display = CANONICAL_TYPE_NAMES.get(lower)
    if display is None:
        bare = lower.removeprefix('system.')
        if bare != lower:
            display = CANONICAL_TYPE_NAMES.get(bare)
    if display is None:
        return None
    has_system = name.lower().startswith('system.')
    display_has_system = display.lower().startswith('system.')
    if has_system and not display_has_system:
        display = F'System.{display}'
    if resolved is not None and (resolved.ranks or resolved.pointers or resolved.byref):
        return str(resolved._replace(name=display, arity=0, arguments=()))
    return display


def canonical_member_name(type_name: str | Ps1TypeName, member: str) -> str | None:
    """
    Return the canonical PascalCase member name for a known .NET type.
    """
    return canonical_member(type_name, member)


def resolve_member_type(
    obj: Expression,
    member: str,
    variable_types: dict[str, Ps1TypeName] | None = None,
) -> Ps1TypeName | None:
    """
    Resolve the .NET result type of accessing `member` on `obj`, or `None` if the object type or the
    member cannot be resolved.
    """
    obj_type = resolve_expression_type(obj, variable_types)
    if obj_type is None:
        return None
    return data_member_type(obj_type, member)


def get_member_order(type_name: str | Ps1TypeName) -> list[str] | None:
    """
    Return the members of a .NET type in PowerShell Get-Member display order: Methods sorted
    alphabetically, then properties sorted alphabetically.
    """
    members = view_members(type_name)
    if members is None:
        return None
    methods = sorted(
        (name for name, m in members.items() if m['kind'] != 'property'),
        key=str.lower,
    )
    properties = sorted(
        (name for name, m in members.items() if m['kind'] == 'property'),
        key=str.lower,
    )
    return methods + properties


def _pipeline_pipes_to_get_member(pipeline: Ps1Pipeline) -> bool:
    """
    Check if the last element in a pipeline is a `Get-Member` command.
    """
    if not pipeline.elements:
        return False
    last = pipeline.elements[-1]
    if not isinstance(last, Ps1PipelineElement):
        return False
    cmd = last.expression
    if not isinstance(cmd, Ps1CommandInvocation):
        return False
    name = get_command_name(cmd)
    return name is not None and name.lower() in GET_MEMBER_ALIASES


def _pipeline_source_type(
    pipeline: Ps1Pipeline,
    variable_types: dict[str, Ps1TypeName] | None = None,
) -> Ps1TypeName | None:
    """
    Determine the .NET type of the expression piped into `Get-Member`. Assumes `Get-Member` is the
    last pipeline element.
    """
    if len(pipeline.elements) < 2:
        return None
    source = pipeline.elements[-2]
    if not isinstance(source, Ps1PipelineElement):
        return None
    if source.expression is None:
        return None
    return resolve_expression_type(source.expression, variable_types)


#: What `foreach` yields from a string: the string itself, not its characters.
_STRING = resolve_type('System.String')


def _resolve_foreach_element_type(iterable: Expression | None) -> Ps1TypeName | None:
    """
    Determine the .NET type of elements produced by a foreach iterable. For a string, PowerShell
    yields the string itself (not individual chars). For an array literal, if all elements share
    the same resolved type, that type is returned.
    """
    if iterable is None:
        return None
    if isinstance(iterable, (Ps1StringLiteral, Ps1HereString)):
        return _STRING
    if isinstance(iterable, Ps1ArrayLiteral) and iterable.elements:
        types = set()
        for elem in iterable.elements:
            if isinstance(elem, Expression):
                t = resolve_expression_type(elem)
                if t is None:
                    return None
                types.add(t)
            else:
                return None
        if len(types) == 1:
            return types.pop()
    return None


def collect_variable_types(root: Node) -> dict[str, Ps1TypeName]:
    """
    Scan the AST for single-assignment variables whose RHS has a resolvable .NET type; e.g.

        $x = New-Object Net.WebClient

    Returns a mapping from lowercase variable name to the canonical `Ps1TypeName`. Mutations that
    do not change the variable's type (member/index assignments, ++/--) are not reassignments.
    """
    assign_counts: dict[str, int] = {}
    typed_assigns: dict[str, Ps1TypeName] = {}
    for var, kind, node in iter_variable_mutations(root):
        if var.scope != Ps1ScopeModifier.NONE:
            continue
        key = var.name.lower()
        if kind in (MutationKind.MEMBER_ASSIGN, MutationKind.INCRDECR):
            continue
        assign_counts[key] = assign_counts.get(key, 0) + 1
        if kind is MutationKind.ASSIGN and isinstance(node, Ps1AssignmentExpression):
            if node.operator == '=' and isinstance(node.value, Expression):
                resolved = resolve_expression_type(node.value)
                if resolved is not None:
                    typed_assigns[key] = resolved
        elif kind is MutationKind.FOREACH and isinstance(node, Ps1ForEachLoop):
            element_type = _resolve_foreach_element_type(node.iterable)
            if element_type is not None:
                typed_assigns[key] = element_type
    return {
        key: type_name
        for key, type_name in typed_assigns.items()
        if assign_counts.get(key, 0) == 1
    }


class VariableTypeAwareTransformer(Transformer):

    def __init__(self):
        super().__init__()
        self._variable_types: dict[str, Ps1TypeName] | None = None

    def visit(self, node: Node):
        if self._variable_types is None:
            self._variable_types = collect_variable_types(node)
        return super().visit(node)


class Ps1TypeSystemSimplifications(VariableTypeAwareTransformer):
    """
    Resolve type-aware patterns. For example, the following resolves to the Nth member name string:

        ($X | Get-Member)[N].Name
    """

    def visit_Ps1MemberAccess(self, node: Ps1MemberAccess):
        self.generic_visit(node)
        result = self._try_resolve_get_member_index_name(node)
        if result is not None:
            return result
        result = self._try_strip_name_on_string(node)
        if result is not None:
            return result
        self._try_normalize_member_case(node)
        return None

    def visit_Ps1InvokeMember(self, node: Ps1InvokeMember):
        self.generic_visit(node)
        self._try_normalize_member_case(node)
        return None

    def _try_normalize_member_case(self, node: Ps1MemberAccess | Ps1InvokeMember):
        if node.object is None:
            return
        obj_type = resolve_expression_type(node.object, self._variable_types)
        if obj_type is None:
            return
        member_name = get_member_name(node.member)
        if member_name is None:
            return
        canonical = canonical_member_name(obj_type, member_name)
        if canonical is not None and canonical != member_name:
            node.member = canonical
            self.mark_changed()

    def _try_strip_name_on_string(
        self,
        node: Ps1MemberAccess,
    ) -> Expression | None:
        """
        Strip `.Name` access on a string literal: After `Where-Object` wildcard resolution or
        `Get-Member` index resolution, a MemberInfo `.Name` access can be left dangling on the
        resolved string: `'GetCmdlets'.Name` -> `'GetCmdlets'`.
        """
        member_name = get_member_name(node.member)
        if member_name is None or member_name.lower() != 'name':
            return None
        if not isinstance(node.object, Ps1StringLiteral):
            return None
        return node.object

    def _try_resolve_get_member_index_name(
        self,
        node: Ps1MemberAccess,
    ) -> Expression | None:
        """
        Resolve ($X | Get-Member)[N].Name to the Nth member name string.
        """
        member_name = get_member_name(node.member)
        if member_name is None or member_name.lower() != 'name':
            return None
        obj = node.object
        if not isinstance(obj, Ps1IndexExpression):
            return None
        if not isinstance(obj.index, Ps1IntegerLiteral):
            return None
        index = obj.index.value
        inner = unwrap_parens(obj.object) if obj.object is not None else None
        if not isinstance(inner, Ps1Pipeline):
            return None
        if not _pipeline_pipes_to_get_member(inner):
            return None
        type_name = _pipeline_source_type(inner, self._variable_types)
        if type_name is None:
            return None
        ordered = get_member_order(type_name)
        if ordered is None or index < 0 or index >= len(ordered):
            return None
        return make_string_literal(ordered[index])
