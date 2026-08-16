"""
Where strict mode comes from, and what it forbids. The parser is fully permissive and always produces
the sloppy-mode parse tree; strict mode never changes how source is parsed, only which already-parsed
constructs are illegal. This module is therefore a pure post-parse pass, and it owns two things.

The first is the vocabulary of the Directive Prologue: which nodes can hold one (`is_prologue_host`),
what a given one holds (`directive_prologue`), whether it declares the Use Strict Directive
(`declares_use_strict`), and which mode any node consequently runs in (`strict_mode_at`). Directive-hood
is a fact about a statement's *position in a statement list*, and a deobfuscator rewrites statement
lists constantly, so every pass that moves, inserts, removes or folds a statement must ask the same
question of the same names — a pass that re-derives the rules is a pass that gets a different answer.

The second is the early errors: `collect_strict_violations` walks a parsed tree, threading strictness
down through function bodies, class bodies and prologues, and records a `StrictViolation` at every
construct a strict region would refuse. The tree is never altered.

The intended consumer is the reflection transform, which inlines payloads from always-sloppy surfaces
(`Function`, indirect `eval`, string timers) and must refuse an inlining that a strict destination would
reject. That wiring is deliberately not part of this module: a payload with no strict violation can still
diverge at runtime, so `collect_strict_violations` is necessary but not sufficient for that decision.
"""
from __future__ import annotations

from dataclasses import dataclass

from refinery.lib.scripts import Expression, Node, Statement
from refinery.lib.scripts.js.lexer import has_legacy_numeric_escape
from refinery.lib.scripts.js.model import (
    JsArrayPattern,
    JsArrowFunctionExpression,
    JsAssignmentExpression,
    JsAssignmentPattern,
    JsBlockStatement,
    JsCatchClause,
    JsClassDeclaration,
    JsClassExpression,
    JsExpressionStatement,
    JsForInStatement,
    JsForOfStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsIfStatement,
    JsImportDefaultSpecifier,
    JsImportNamespaceSpecifier,
    JsImportSpecifier,
    JsLabeledStatement,
    JsMemberExpression,
    JsMethodDefinition,
    JsNumericLiteral,
    JsObjectPattern,
    JsProperty,
    JsPropertyDefinition,
    JsRestElement,
    JsScript,
    JsStaticBlock,
    JsStringLiteral,
    JsUnaryExpression,
    JsUpdateExpression,
    JsVariableDeclaration,
    JsVariableDeclarator,
    JsVarKind,
    JsWithStatement,
    strip_parens,
)


@dataclass(frozen=True)
class StrictViolation:
    """
    A single strict-mode early error found in an otherwise sloppy-parsed tree. `rule` is a stable slug
    naming the violated restriction; `name` carries the offending identifier for the name-based rules and
    is empty otherwise. A violation only records that the code at `offset` would be a `SyntaxError` if its
    enclosing region ran in strict mode; the parse tree is never changed.
    """
    offset: int
    rule: str
    name: str = ''


def is_leading_zero_number(raw: str) -> bool:
    return len(raw) >= 2 and raw[0] == '0' and raw[1] in '0123456789'


def has_octal_string_escape(node: JsStringLiteral) -> bool:
    """
    Whether a string literal was written with an escape that strict code rejects. It is the same
    spelling a template excludes from its grammar, so the scan itself lives beside the escapes it
    reads and both rules ask it there.
    """
    return has_legacy_numeric_escape(node.body)


def is_use_strict(node: JsStringLiteral) -> bool:
    """
    Whether a literal spells the Use Strict Directive. It is asked of the spelling rather than of
    the value, because a directive is one: a literal that denotes `use strict` through an escape is
    not the directive, and neither is one the source never closed.
    """
    return node.terminated and node.body == 'use strict'


def spelling_states(body: str) -> tuple[bool, bool]:
    """
    What a literal's spelling states, as against what it denotes: whether it is the Use Strict
    Directive, and whether it carries an escape strict code rejects. Both are facts about how the
    literal was written and about nothing else, so a pass that re-spells one may do so only where
    neither answer moves — re-spelling `'use\\x20strict'` as `'use strict'` writes a directive the
    source never wrote, and every line behind it becomes strict code.
    """
    return body == 'use strict', has_legacy_numeric_escape(body)


_FUNCTION_NODES = (JsFunctionDeclaration, JsFunctionExpression, JsArrowFunctionExpression)

_FunctionNode = JsFunctionDeclaration | JsFunctionExpression | JsArrowFunctionExpression


def _statement_list(node: Node | None) -> list[Statement] | None:
    """
    The statement list *node* holds directly, or `None` when it holds none. Only the three node types
    that can host a Directive Prologue are answered for, so a caller that already knows it is looking
    at a host reads the list here without consulting the tree above it.
    """
    if isinstance(node, (JsScript, JsBlockStatement, JsStaticBlock)):
        return node.body
    return None


def is_prologue_host(node: Node | None) -> bool:
    """
    Whether *node* holds a statement list that a Directive Prologue can open (§11.2.1): a script body,
    a function body, or a class static block. Nothing else does. A plain block, the body of a `try`,
    `catch` or `finally`, a labelled statement, a `switch` case and the expression body of a concise
    arrow all hold code no directive governs, so a `'use strict'` written at the head of one is an
    ordinary string-valued statement that changes no mode.

    A function body is recognized through the function that owns it, because a body and a plain block
    are the same node type and only the tree above tells them apart.
    """
    if isinstance(node, (JsScript, JsStaticBlock)):
        return True
    if isinstance(node, JsBlockStatement):
        owner = node.parent
        return isinstance(owner, _FUNCTION_NODES) and owner.body is node
    return False


def directive_prologue(host: Node | None) -> list[JsExpressionStatement]:
    """
    The Directive Prologue of *host*: the run of statements it opens with that consist of nothing but
    a string literal. The run ends at the first statement that is anything else, so it is a prefix, and
    every statement behind that one is ordinary code however it happens to be spelled.

    A parenthesized literal is not one of them. A directive is a statement whose expression *is* the
    literal, so `('use strict');` states nothing, and the parser keeps the parenthesis as a node of its
    own precisely so that this stays decidable.

    *host* is taken to be a prologue host; where a caller must find the host from a statement inside
    it, `is_prologue_host` decides that.
    """
    prologue: list[JsExpressionStatement] = []
    for statement in _statement_list(host) or ():
        if not isinstance(statement, JsExpressionStatement):
            break
        if not isinstance(statement.expression, JsStringLiteral):
            break
        prologue.append(statement)
    return prologue


def declares_use_strict(host: Node | None) -> bool:
    """
    Whether the Directive Prologue of *host* holds the Use Strict Directive, which makes the code
    *host* encloses strict. The directive need not open the prologue: every string-literal statement
    ahead of it is a directive too, and one the language does not recognize is simply inert.
    """
    for statement in directive_prologue(host):
        expression = statement.expression
        if isinstance(expression, JsStringLiteral) and is_use_strict(expression):
            return True
    return False


def joins_directive_prologue(statement: Statement) -> bool:
    """
    Whether *statement* would enter the Directive Prologue of the body that holds it were it spelled as
    a string literal: it sits in a prologue host, and nothing but string-literal statements precede it.
    A pass that rewrites such a statement into a literal hands the prologue that statement *and* every
    string-literal statement standing behind it, so a `'use strict'` that was ordinary code becomes the
    directive that makes the whole body strict.
    """
    host = statement.parent
    body = _statement_list(host)
    if body is None or not is_prologue_host(host):
        return False
    index = len(directive_prologue(host))
    return index < len(body) and body[index] is statement


def strict_mode_at(node: Node) -> bool:
    """
    Whether the code at *node* runs in strict mode. Mode is inherited (§11.2.2): a body is strict when
    its own Directive Prologue declares it or when the code enclosing it is strict, and every part of a
    class definition is strict whatever encloses it (§15.7). *node* itself counts, so asking this of a
    function body answers the mode that body runs in.

    A function's directive reaches further than the body that holds it: the parameter list and the name
    the function binds are strict code too, which is why `function f(eval) { 'use strict'; }` is refused
    and `function f(eval) {}` is a program. Neither stands inside the body, so the whole function is
    asked, not only the host.

    Module code is strict as well; that is not decided here, because module-ness is a fact about the
    whole program rather than about any node in it.
    """
    cursor: Node | None = node
    while cursor is not None:
        if isinstance(cursor, (JsClassDeclaration, JsClassExpression)):
            return True
        if is_prologue_host(cursor) and declares_use_strict(cursor):
            return True
        if isinstance(cursor, _FUNCTION_NODES) and declares_use_strict(cursor.body):
            return True
        cursor = cursor.parent
    return False


def has_simple_parameters(fn: _FunctionNode) -> bool:
    """
    Whether *fn* has a simple parameter list (§15.1.3): every parameter is a plain identifier, with no
    default, no rest element and no destructuring. An empty list is simple — nothing in it is anything
    else — which is what makes a Use Strict Directive legal in `function f() { 'use strict'; }`.

    A rule that additionally needs there to be *something* to be simple about must ask that separately.
    Whether the `arguments` object aliases a parameter is such a rule: with no parameters there is
    nothing to alias, but the parameter list is simple all the same.
    """
    return all(isinstance(param, JsIdentifier) for param in fn.params)


_STRICT_RESERVED = frozenset({
    'implements',
    'interface',
    'let',
    'package',
    'private',
    'protected',
    'public',
    'static',
    'yield',
})

_EVAL_ARGS = frozenset({'eval', 'arguments'})


def _child_strictness(node: Node, strict: bool) -> bool:
    if isinstance(node, JsScript):
        return strict or declares_use_strict(node)
    if isinstance(node, (JsClassDeclaration, JsClassExpression)):
        return True
    if not isinstance(node, _FUNCTION_NODES):
        return strict
    body = node.body
    if isinstance(body, JsBlockStatement):
        return strict or declares_use_strict(body)
    return strict


def _record_nested_function(stmt: Statement | None, out: list[StrictViolation]) -> None:
    if isinstance(stmt, JsFunctionDeclaration):
        out.append(StrictViolation(stmt.offset, 'function-in-statement'))


def _check_node(node: Node, strict: bool, out: list[StrictViolation]) -> None:
    if not strict:
        return
    if isinstance(node, JsNumericLiteral):
        if is_leading_zero_number(node.raw):
            out.append(StrictViolation(node.offset, 'octal-literal'))
    elif isinstance(node, JsStringLiteral):
        if has_octal_string_escape(node):
            out.append(StrictViolation(node.offset, 'octal-escape'))
    elif isinstance(node, JsWithStatement):
        out.append(StrictViolation(node.offset, 'with-statement'))
    elif isinstance(node, JsUnaryExpression):
        if node.operator == 'delete':
            target = strip_parens(node.operand)
            if isinstance(target, JsIdentifier) and target.name != 'super':
                out.append(StrictViolation(node.offset, 'delete-of-reference'))
    elif isinstance(node, JsIfStatement):
        _record_nested_function(node.consequent, out)
        _record_nested_function(node.alternate, out)
    elif isinstance(node, JsLabeledStatement):
        _record_nested_function(node.body, out)
    elif isinstance(node, JsForInStatement):
        left = node.left
        if isinstance(left, JsVariableDeclaration) and left.kind is JsVarKind.VAR:
            declarations = left.declarations
            if len(declarations) == 1 and declarations[0].init is not None:
                out.append(StrictViolation(left.offset, 'for-in-var-init'))


def _target_identifiers(target: Node | None) -> list[JsIdentifier]:
    """
    Every identifier bound or assigned by a binding or assignment target, flattening array and object
    patterns, defaults, and rest elements down to their leaves. A pattern default value and a computed
    property key are references rather than targets, so they are left for the ordinary traversal; only
    the names actually bound by the pattern are returned.
    """
    result: list[JsIdentifier] = []
    stack: list[Node | None] = [target]
    while stack:
        node = stack.pop()
        if isinstance(node, JsIdentifier):
            result.append(node)
        elif isinstance(node, JsArrayPattern):
            stack.extend(node.elements)
        elif isinstance(node, JsObjectPattern):
            for prop in node.properties:
                if isinstance(prop, JsProperty):
                    stack.append(prop.value)
                elif isinstance(prop, JsRestElement):
                    stack.append(prop.argument)
        elif isinstance(node, JsAssignmentPattern):
            stack.append(node.left)
        elif isinstance(node, JsRestElement):
            stack.append(node.argument)
    return result


def _is_property_name_position(node: JsIdentifier) -> bool:
    parent = node.parent
    if isinstance(parent, JsMemberExpression):
        return parent.property is node and not parent.computed
    if isinstance(parent, (JsProperty, JsMethodDefinition, JsPropertyDefinition)):
        return parent.key is node and not parent.computed
    return False


def _flag_name(ident: JsIdentifier, out: list[StrictViolation]) -> None:
    if ident.name in _EVAL_ARGS:
        out.append(StrictViolation(ident.offset, 'eval-arguments-target', ident.name))
    elif ident.name in _STRICT_RESERVED:
        out.append(StrictViolation(ident.offset, 'reserved-word', ident.name))


def _flag_bound(target: Node | None, strict: bool, out: list[StrictViolation], handled: set[int]) -> None:
    for ident in _target_identifiers(target):
        handled.add(id(ident))
        if strict:
            _flag_name(ident, out)


def _check_parameters(
    params: list[Expression],
    strict: bool,
    out: list[StrictViolation],
    handled: set[int],
) -> None:
    seen: set[str] = set()
    for param in params:
        for ident in _target_identifiers(param):
            handled.add(id(ident))
            if not strict:
                continue
            _flag_name(ident, out)
            if ident.name in seen:
                out.append(StrictViolation(ident.offset, 'duplicate-parameter', ident.name))
            else:
                seen.add(ident.name)


def _check_names(
    node: Node,
    cur_strict: bool,
    child_strict: bool,
    out: list[StrictViolation],
    handled: set[int],
) -> None:
    if isinstance(node, (JsFunctionDeclaration, JsFunctionExpression)):
        _check_parameters(node.params, child_strict, out, handled)
        _flag_bound(node.id, child_strict, out, handled)
    elif isinstance(node, JsArrowFunctionExpression):
        _check_parameters(node.params, child_strict, out, handled)
    elif isinstance(node, (JsClassDeclaration, JsClassExpression)):
        _flag_bound(node.id, child_strict, out, handled)
    elif isinstance(node, JsVariableDeclarator):
        _flag_bound(node.id, cur_strict, out, handled)
    elif isinstance(node, JsCatchClause):
        _flag_bound(node.param, cur_strict, out, handled)
    elif isinstance(node, (JsImportSpecifier, JsImportDefaultSpecifier, JsImportNamespaceSpecifier)):
        _flag_bound(node.local, cur_strict, out, handled)
    elif isinstance(node, JsAssignmentExpression):
        _flag_bound(node.left, cur_strict, out, handled)
    elif isinstance(node, JsUpdateExpression):
        _flag_bound(node.argument, cur_strict, out, handled)
    elif isinstance(node, (JsForInStatement, JsForOfStatement)):
        if not isinstance(node.left, JsVariableDeclaration):
            _flag_bound(node.left, cur_strict, out, handled)
    elif isinstance(node, JsIdentifier):
        if (
            id(node) not in handled
            and cur_strict
            and node.name in _STRICT_RESERVED
            and not _is_property_name_position(node)
        ):
            out.append(StrictViolation(node.offset, 'reserved-word', node.name))


def collect_strict_violations(node: Node, *, strict: bool = False) -> list[StrictViolation]:
    """
    Every strict-mode early error in the tree rooted at *node*, in source order. *strict* seeds the
    strictness of *node* itself; the pass then forces strict inside class bodies and inside any function
    whose body opens with a `"use strict"` directive, so a violation is recorded even when the seed is
    sloppy but the offending code sits in an inherently strict region. An empty result means the tree has
    no strict-mode parse error; it does not imply the tree behaves identically in strict mode, since some
    divergences surface only at runtime.
    """
    out: list[StrictViolation] = []
    handled: set[int] = set()
    stack: list[tuple[Node, bool]] = [(node, strict)]
    while stack:
        current, current_strict = stack.pop()
        child_strict = _child_strictness(current, current_strict)
        _check_node(current, current_strict, out)
        _check_names(current, current_strict, child_strict, out, handled)
        for child in current.children():
            stack.append((child, child_strict))
    out.sort(key=lambda violation: violation.offset)
    return out
