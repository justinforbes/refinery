"""
The effect layer of the PowerShell analysis substrate: whether evaluating a node produces an
observable side effect, and what a standalone statement contributes to the body it sits in. Every
pass that decides "is it safe to delete this?" asks here, so that no two of them can disagree.

These are free functions rather than a model class because the facts they compute are syntactic: a
conservative allow-list over one expression, needing no information from anywhere else in the tree.
A cached model arrives with the first genuine summary fact — interprocedural purity, which has to be
computed over the `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel`.

**Scope.** `StatementEffect` models emission and side effect, not *fault* behavior: it has no member
for a statement that may throw. The trap and try/catch passes therefore keep statement predicates of
their own for reasoning about exceptions, and folding those into `statement_effect` requires
deciding a fault semantics first — it is not a simplification that can be made silently.
"""
from __future__ import annotations

import enum

from typing import Sequence

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.ast import (
    extract_new_object,
    get_body,
    get_command_name,
    is_builtin_variable,
    normalize_dotnet_type_name,
)
from refinery.lib.scripts.ps1.model import (
    Ps1AccessKind,
    Ps1ArrayExpression,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1BinaryExpression,
    Ps1CastExpression,
    Ps1CommandArgument,
    Ps1CommandInvocation,
    Ps1ExpandableString,
    Ps1ExpressionStatement,
    Ps1FunctionDefinition,
    Ps1HashLiteral,
    Ps1HereString,
    Ps1IndexExpression,
    Ps1IntegerLiteral,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1ParenExpression,
    Ps1Pipeline,
    Ps1PipelineElement,
    Ps1RangeExpression,
    Ps1RealLiteral,
    Ps1Script,
    Ps1ScriptBlock,
    Ps1StringLiteral,
    Ps1SubExpression,
    Ps1TypeExpression,
    Ps1UnaryExpression,
    Ps1Variable,
)

_PURE_STATIC_TYPES = frozenset({
    'array',
    'bitconverter',
    'char',
    'collections.arraylist',
    'collections.generic.dictionary',
    'collections.generic.hashset',
    'collections.generic.list',
    'collections.hashtable',
    'convert',
    'datetime',
    'decimal',
    'double',
    'environment',
    'guid',
    'int',
    'int32',
    'int64',
    'io.path',
    'ipaddress',
    'math',
    'object',
    'security.securestring',
    'securestring',
    'string',
    'text.stringbuilder',
    'timespan',
    'version',
})

_PURE_STATIC_METHODS = frozenset({
    ('diagnostics.process', 'getcurrentprocess'),
    ('threading.tasks.task', 'delay'),
    ('collections.hashtable', 'synchronized'),
})


def _pure_type_name(name: str) -> str:
    """
    Normalize a .NET type name for purity lookup: lower-cased, `System.` prefix removed, and any
    generic-argument suffix (`[byte]` or the arity marker before it) stripped, so that both

        System.Collections.Generic.List
        List[byte]

    reduce to the same `collections.generic.list` key.
    """
    name = normalize_dotnet_type_name(name)
    for separator in ('[', '`'):
        name = name.split(separator, 1)[0]
    return name


_PURE_INSTANCE_METHODS = frozenset({
    'adddays',
    'addhours',
    'addminutes',
    'addmonths',
    'addseconds',
    'addyears',
    'compareto',
    'contains',
    'endswith',
    'equals',
    'gethashcode',
    'gettype',
    'indexof',
    'lastindexof',
    'length',
    'padleft',
    'padright',
    'split',
    'startswith',
    'substring',
    'tochar',
    'tochararray',
    'tolower',
    'tostring',
    'touniversaltime',
    'toupper',
    'trim',
    'trimend',
    'trimstart',
})

_PURE_CMDLETS = frozenset({
    'get-childitem',
    'get-command',
    'get-content',
    'get-date',
    'get-item',
    'get-location',
    'get-process',
    'get-random',
    'get-variable',
    'measure-object',
    'out-null',
    'out-string',
    'select-object',
    'sort-object',
    'where-object',
})

_PURE_PIPELINE_CMDLETS = frozenset({
    'foreach-object',
    'select-object',
    'sort-object',
    'where-object',
})


def _command_body_is_pure(cmd: Ps1CommandInvocation) -> bool:
    """
    Check whether all script block arguments of a pipeline cmdlet (ForEach-Object, Where-Object,
    etc.) have side-effect-free bodies. These cmdlets are pure transforms: they evaluate a script
    block per input item without mutating state themselves. Note: the `$Null = <pure>` discard
    idiom is NOT currently recognized here because `is_side_effect_free` has no case for
    `Ps1AssignmentExpression`; such bodies are caught at statement level by
    `pipeline_ends_with_void_foreach` instead.
    """
    # TODO: teach `is_side_effect_free` to recognize `$Null = <pure>` assignments as pure so that
    # this function correctly handles ForEach bodies containing the discard idiom without relying
    # on the separate `pipeline_ends_with_void_foreach` path.
    for arg in cmd.arguments:
        block = arg.value if isinstance(arg, Ps1CommandArgument) else arg
        if not isinstance(block, Ps1ScriptBlock):
            continue
        for stmt in block.body:
            if not isinstance(stmt, Ps1ExpressionStatement):
                return False
            if stmt.expression is not None and not is_side_effect_free(stmt.expression):
                return False
    return True


def is_side_effect_free(node) -> bool:
    """
    Conservative check: return `True` only when evaluating `node` is guaranteed to produce no
    observable side effects beyond yielding a value.
    """
    if isinstance(node, (Ps1StringLiteral, Ps1HereString, Ps1IntegerLiteral, Ps1RealLiteral)):
        return True
    if isinstance(node, Ps1TypeExpression):
        return True
    if isinstance(node, Ps1Variable):
        return True
    if isinstance(node, Ps1ParenExpression):
        return node.expression is None or is_side_effect_free(node.expression)
    if isinstance(node, Ps1CastExpression):
        return is_side_effect_free(node.operand)
    if isinstance(node, Ps1UnaryExpression):
        if node.operator in ('++', '--'):
            return False
        return is_side_effect_free(node.operand)
    if isinstance(node, Ps1BinaryExpression):
        return is_side_effect_free(node.left) and is_side_effect_free(node.right)
    if isinstance(node, Ps1RangeExpression):
        return is_side_effect_free(node.start) and is_side_effect_free(node.end)
    if isinstance(node, Ps1ArrayLiteral):
        return all(is_side_effect_free(e) for e in node.elements)
    if isinstance(node, Ps1HashLiteral):
        return all(is_side_effect_free(value) for _key, value in node.pairs)
    if isinstance(node, Ps1ArrayExpression):
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, Ps1ExpressionStatement) and stmt.expression is not None:
                return is_side_effect_free(stmt.expression)
        return len(node.body) == 0
    if isinstance(node, Ps1IndexExpression):
        return is_side_effect_free(node.object) and is_side_effect_free(node.index)
    if isinstance(node, Ps1MemberAccess):
        return is_side_effect_free(node.object)
    if isinstance(node, Ps1InvokeMember):
        if not all(is_side_effect_free(a) for a in node.arguments):
            return False
        if node.access == Ps1AccessKind.STATIC:
            obj = node.object
            if isinstance(obj, Ps1TypeExpression):
                type_name = _pure_type_name(obj.name)
                if type_name in _PURE_STATIC_TYPES:
                    return True
                member = node.member
                if (
                    isinstance(member, str)
                    and (type_name, member.lower()) in _PURE_STATIC_METHODS
                ):
                    return True
        elif is_side_effect_free(node.object):
            member = node.member
            if isinstance(member, str) and member.lower() in _PURE_INSTANCE_METHODS:
                return True
        return False
    if isinstance(node, Ps1CommandInvocation):
        new_object = extract_new_object(node)
        if new_object is not None:
            type_name, ctor_args = new_object
            if _pure_type_name(type_name) in _PURE_STATIC_TYPES:
                return all(is_side_effect_free(a) for a in ctor_args)
            return False
        name = get_command_name(node)
        if name is None:
            return False
        if name.lower() in _PURE_CMDLETS:
            return True
        if name.lower() in _PURE_PIPELINE_CMDLETS:
            return _command_body_is_pure(node)
        return False
    if isinstance(node, Ps1Pipeline):
        return all(
            isinstance(el, Ps1PipelineElement) and is_side_effect_free(el.expression)
            for el in node.elements
        )
    if isinstance(node, Ps1ExpandableString):
        return all(is_side_effect_free(p) for p in node.parts)
    return False


def is_pure_constant(node) -> bool:
    """
    Whether an expression is a side-effect-free constant that can be removed as a standalone
    statement: a numeric literal or one of the built-in constants `$Null`, `$True`, `$False`,
    through any enclosing parentheses and unary sign. String literals are excluded because they may
    be intentional pipeline output.

    This is a strict refinement of `StatementEffect.OUTPUT`: an expression statement whose
    expression is a pure constant always classifies as `OUTPUT`. The two pruning passes therefore
    have nested candidate sets rather than independently drifting ones — the dead-code pass, which
    prunes only constants, is provably the more conservative of the two.
    """
    if isinstance(node, (Ps1IntegerLiteral, Ps1RealLiteral)):
        return True
    if is_builtin_variable(node):
        return True
    if isinstance(node, Ps1ParenExpression):
        return is_pure_constant(node.expression)
    if isinstance(node, Ps1UnaryExpression) and node.operator in ('+', '-'):
        return is_pure_constant(node.operand)
    return False


class StatementEffect(enum.Enum):
    """
    The observable effect of evaluating a standalone statement, used by every pass that decides
    whether a statement can be pruned from a body:

    - `EFFECT`: the statement performs a side effect (a command call, a store to a real variable, an
      increment); it must be preserved.
    - `OUTPUT`: the statement is side-effect-free but yields a value to the enclosing pipeline (a
      bare constant, a pure expression); it is junk at a discarding position, but in a captured body
      it may be the return value, so removing it needs an emit-safety check.
    - `DISCARD`: the statement is a syntactic no-op that yields nothing and does nothing observable
      (an empty statement, the `$Null = <pure>` discard idiom, a `[Void]` cast, a `... | Out-Null`
      pipeline, a discarding `ForEach`); it is always safe to remove, even when it empties the body.
    """
    EFFECT = 'effect'
    OUTPUT = 'output'
    DISCARD = 'discard'


def statement_effect(stmt) -> StatementEffect:
    """
    Classify the observable effect of a standalone statement as a `StatementEffect`. This is the one
    shared authority the dead-code and junk-removal passes consult so they never disagree about
    whether a statement carries a body's output: a `DISCARD` emits nothing and can always be
    dropped, an `OUTPUT` yields a value that emit-safety must protect in a captured body, and an
    `EFFECT` must always be kept.
    """
    if not isinstance(stmt, Ps1ExpressionStatement):
        return StatementEffect.EFFECT
    expr = stmt.expression
    if expr is None:
        return StatementEffect.DISCARD
    if isinstance(expr, Ps1CastExpression) and expr.type_name.lower() == 'void':
        return StatementEffect.DISCARD
    if isinstance(expr, Ps1Pipeline):
        if pipeline_ends_with_out_null(expr) and pipeline_prefix_is_pure(expr):
            return StatementEffect.DISCARD
        if pipeline_ends_with_void_foreach(expr) and pipeline_prefix_is_pure(expr):
            return StatementEffect.DISCARD
        if pipeline_ends_with_cmdlet(expr, _PURE_PIPELINE_CMDLETS):
            # A pure pipeline cmdlet (`... | Where-Object {...}`) yields a filtered value a caller
            # may consume, so it is kept even though it performs no side effect of its own.
            return StatementEffect.EFFECT
    if (
        isinstance(expr, Ps1AssignmentExpression)
        and expr.operator == '='
        and is_builtin_variable(expr.target, {'null'})
    ):
        if expr.value is not None and is_side_effect_free(expr.value):
            return StatementEffect.DISCARD
        return StatementEffect.EFFECT
    if is_side_effect_free(expr):
        return StatementEffect.OUTPUT
    return StatementEffect.EFFECT


def pipeline_ends_with_out_null(pipeline: Ps1Pipeline) -> bool:
    if len(pipeline.elements) < 2:
        return False
    last = pipeline.elements[-1]
    if not isinstance(last, Ps1PipelineElement):
        return False
    expr = last.expression
    if isinstance(expr, Ps1CommandInvocation):
        name = get_command_name(expr)
        return name is not None and name.lower() == 'out-null'
    return False


def pipeline_prefix_is_pure(pipeline: Ps1Pipeline) -> bool:
    for el in pipeline.elements[:-1]:
        if not isinstance(el, Ps1PipelineElement):
            return False
        if not is_side_effect_free(el.expression):
            return False
    return True


def pipeline_ends_with_void_foreach(pipeline: Ps1Pipeline) -> bool:
    """
    Detect junk pipelines like `... | ForEach-Object { [Void]$_ }` or
    `... | ForEach-Object { $Null = $_ }` where the ForEach body explicitly discards all output.
    These are anti-analysis noise injected into malware scripts.
    """
    if len(pipeline.elements) < 2:
        return False
    last = pipeline.elements[-1]
    if not isinstance(last, Ps1PipelineElement):
        return False
    expr = last.expression
    if not isinstance(expr, Ps1CommandInvocation):
        return False
    name = get_command_name(expr)
    if name is None or name.lower() != 'foreach-object':
        return False
    for arg in expr.arguments:
        block = arg.value if isinstance(arg, Ps1CommandArgument) else arg
        if not isinstance(block, Ps1ScriptBlock):
            continue
        for stmt in block.body:
            if not isinstance(stmt, Ps1ExpressionStatement) or stmt.expression is None:
                return False
            ex = stmt.expression
            if isinstance(ex, Ps1CastExpression) and ex.type_name.lower() == 'void':
                continue
            if (
                isinstance(ex, Ps1AssignmentExpression)
                and ex.operator == '='
                and is_builtin_variable(ex.target, {'null'})
                and (ex.value is None or is_side_effect_free(ex.value))
            ):
                continue
            return False
    return True


def pipeline_ends_with_cmdlet(pipeline: Ps1Pipeline, names: frozenset) -> bool:
    if len(pipeline.elements) < 2:
        return False
    last = pipeline.elements[-1]
    if not isinstance(last, Ps1PipelineElement):
        return False
    expr = last.expression
    if not isinstance(expr, Ps1CommandInvocation):
        return False
    name = get_command_name(expr)
    return name is not None and name.lower() in names


class BodyRole(enum.Enum):
    """
    How a statement body relates to the code around it — the emission question every pruning pass
    has to answer before it removes anything. A `refinery.lib.scripts.Block` or
    `refinery.lib.scripts.ps1.model.Ps1Code` body is one of:

    - `OPAQUE`: the body's value is captured (an assignment right-hand side, `$(...)`, `@(...)`, a
      stored or argument scriptblock, a piped `&{}`); pruning any statement could destroy an
      observable value, so the body is left untouched.
    - `SCRIPT`: the script root. It has no return value — its output goes to the host — but it must
      never be pruned away entirely, which is what `pruning_erases_body` guards.
    - `RETURNING`: a body whose value the caller observes — a function or method body, or a bare
      `&{ ... }` / `.{ ... }` in statement position. Removing the statement that carries the output
      silences the return value, so pruning goes through `output_observed` and `output_is_covered`.
    - `NESTED`: a plain nested block that runs for its side effects (a loop or `if` body in
      statement position); it has no observable value of its own, so statements may be pruned
      freely.
    """
    OPAQUE = 'opaque'
    SCRIPT = 'script'
    RETURNING = 'returning'
    NESTED = 'nested'


def _scriptblock_is_captured(block: Ps1ScriptBlock) -> bool:
    """
    Return `True` when the value of a `refinery.lib.scripts.ps1.model.Ps1ScriptBlock` is captured
    rather than run for its observable output. A bare `&{ ... }` / `.{ ... }` in statement position
    produces output that the pass may prune into; every other scriptblock (a stored closure
    `$x = { ... }`, an argument block, or an invocation whose result is assigned, passed, or piped)
    is treated as captured and left opaque.
    """
    parent = block.parent
    if isinstance(parent, Ps1FunctionDefinition):
        return False
    if not (isinstance(parent, Ps1CommandInvocation) and parent.name is block):
        return True
    invocation_parent = parent.parent
    if isinstance(invocation_parent, Ps1ExpressionStatement):
        return False
    if isinstance(invocation_parent, Ps1PipelineElement):
        pipeline = invocation_parent.parent
        if (
            isinstance(pipeline, Ps1Pipeline)
            and len(pipeline.elements) == 1
            and isinstance(pipeline.parent, Ps1ExpressionStatement)
        ):
            return False
    return True


def body_role(node) -> BodyRole | None:
    """
    Classify the statement body that `node` owns as a `BodyRole`, or return `None` when `node` owns
    no prunable body — which is also how `@( ... )` stays out of every pruning walk, since
    `refinery.lib.scripts.ps1.ast.get_body` deliberately does not recognize it. Ambiguous capture
    always resolves to `OPAQUE`.

    A plain `refinery.lib.scripts.Block` — a loop, `if`, `try`, `catch`, `finally`, or `trap` body —
    carries no role of its own and derives one by walking outward to the nearest body owner. That
    walk reports the *owner's* role only for a function body, so the same block classifies three
    ways depending on where it sits:

        if ($x) { 1 }                    at script level  ->  NESTED
        function f { if ($x) { 1 } }                      ->  RETURNING
        &{ if ($x) { 1 } }                                ->  NESTED

    A nested block's value is observed exactly when its owner's is, so the consistent answer would
    be the owner's role in all three cases, and `NESTED` is the more permissive one at both the
    script and the `&{}` boundary. The passes have shipped with this behavior and all three traces
    are pinned by test; resolving it needs the reachability of the flow layer, so it is deliberately
    left as it stands rather than changed as a side effect of consolidating the authority here.
    """
    if get_body(node) is None:
        return None
    if isinstance(node, Ps1Script):
        return BodyRole.SCRIPT
    if isinstance(node, Ps1SubExpression):
        return BodyRole.OPAQUE
    if isinstance(node, Ps1ScriptBlock):
        if isinstance(node.parent, Ps1FunctionDefinition) and node.parent.body is node:
            return BodyRole.RETURNING
        return BodyRole.OPAQUE if _scriptblock_is_captured(node) else BodyRole.RETURNING
    prev = node
    cursor = node.parent
    while cursor is not None:
        if isinstance(cursor, (Ps1SubExpression, Ps1ArrayExpression)):
            return BodyRole.OPAQUE
        if isinstance(cursor, Ps1AssignmentExpression) and cursor.value is prev:
            return BodyRole.OPAQUE
        if isinstance(cursor, Ps1ScriptBlock):
            if _scriptblock_is_captured(cursor):
                return BodyRole.OPAQUE
            if isinstance(cursor.parent, Ps1FunctionDefinition) and cursor.parent.body is cursor:
                return BodyRole.RETURNING
            return BodyRole.NESTED
        if isinstance(cursor, Ps1Script):
            return BodyRole.NESTED
        prev = cursor
        cursor = cursor.parent
    return BodyRole.NESTED


def output_observed(role: BodyRole) -> bool:
    """
    Whether a body of this role has a return value that pruning must protect. True only for
    `BodyRole.RETURNING`: a `NESTED` body has no observable value, the `SCRIPT` root has no return
    value, and an `OPAQUE` body is never pruned at all.
    """
    return role is BodyRole.RETURNING


def output_is_covered(survivors: Sequence[Node]) -> bool:
    """
    Whether some statement in `survivors` still carries the body's output, so that removing the
    pure-output statements around it cannot silence a `BodyRole.RETURNING` body's return value.

    `survivors` is the surviving statement set itself and never a node to walk up from. A caller may
    hold freshly synthesized statements that are not parented into a body yet, and statements
    hoisted out of a pruned block still point at the block they came from; answering this question
    by walking `parent` is what used to delete live return values.

    The check is coarse: every survivor that is not a function definition counts as covering,
    including a conditional that may not execute and a statement that emits nothing at all. It
    therefore over-counts, permitting a prune that a precise analysis would refuse. This is the
    semantics the junk pass has shipped with; tightening it needs reachability.
    """
    return any(not isinstance(stmt, Ps1FunctionDefinition) for stmt in survivors)


def pruning_erases_body(role: BodyRole, survivors: Sequence[Node]) -> bool:
    """
    Whether pruning a body of this role down to `survivors` would erase it: nothing would survive,
    and a body of this role must not become empty. Only the `BodyRole.SCRIPT` root qualifies — a
    script that is nothing but function definitions is a module whose functions may be dot-sourced,
    and a script that is nothing but `42` still emits `42` — so emptying it would delete real code.
    Every other role may legitimately prune to nothing; that is what turns an injected junk function
    inert.

    Like `output_is_covered`, this takes the surviving statement set itself and never walks up from
    a node.
    """
    return not survivors and role is BodyRole.SCRIPT


def body_is_inert(node) -> bool:
    """
    Whether the body that `node` owns neither emits a value nor performs a side effect: `node` owns
    no body at all, the body is empty, or every statement in it is a `StatementEffect.DISCARD`. An
    inert function body makes the function itself unobservable, so its definition and its bare call
    sites can be dropped together.
    """
    body = get_body(node)
    if body is None:
        return True
    return all(statement_effect(stmt) is StatementEffect.DISCARD for stmt in body)
