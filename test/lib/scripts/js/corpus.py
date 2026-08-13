"""
The JavaScript the js tests are quantified over, and what the language says about the shape of a
tree that spells it.

Everything here is hand-authored and nothing is read from disk, downloaded, or derived from a
sample. The snippets were checked against a real engine before they were written down — V8 was
asked to compile each one, which parses without running anything — so that a snippet is JavaScript
because an engine says so and not because our own parser accepted it. The one entry V8 refuses is
`JsDecorator`: decorators are a proposal that has not shipped, and the model reads them anyway.

Nothing in this module may ever be executed. Compiling a snippet is not running it, and the
fidelity harness only ever parses and prints.
"""
from __future__ import annotations

from refinery.lib.scripts import Node
from refinery.lib.scripts.js.model import JsTemplateLiteral

#: One program per node class, written by hand rather than generated, and kept minimal so that a
#: failure names the construct it is about. Where a node has a child list, the snippet fills it with
#: two entries: the fidelity generator truncates from there, and a rendering that is only correct at
#: the cardinality its author had in mind is what that is looking for.
SNIPPETS: dict[str, str] = {
    'JsArrayExpression'          : '[1, 2];',
    'JsArrayPattern'             : 'var [a, b] = c;',
    'JsArrowFunctionExpression'  : 'var f = (a, b) => a + b;',
    'JsAssignmentExpression'     : 'a = 1;',
    'JsAssignmentPattern'        : 'function f(a = 1, b = 2) { return a; }',
    'JsAwaitExpression'          : 'async function f(p) { return await p; }',
    'JsBigIntLiteral'            : 'var x = 10n;',
    'JsBinaryExpression'         : '1 + 2;',
    'JsBlockStatement'           : '{ a; b; }',
    'JsBooleanLiteral'           : 'var x = true;',
    'JsBreakStatement'           : 'while (a) { break; }',
    'JsCallExpression'           : 'f(1, 2);',
    'JsCatchClause'              : 'try { a; } catch (e) { b; }',
    'JsClassBody'                : 'class C { m() {} n() {} }',
    'JsClassDeclaration'         : 'class C extends B { m() {} }',
    'JsClassExpression'          : 'var C = class extends B { m() {} };',
    'JsConditionalExpression'    : 'a ? b : c;',
    'JsContinueStatement'        : 'while (a) { continue; }',
    'JsDebuggerStatement'        : 'debugger;',
    'JsDecorator'                : 'class C { @a @b m() {} }',
    'JsDoWhileStatement'         : 'do { a; } while (b);',
    'JsEmptyStatement'           : ';',
    'JsExportAllDeclaration'     : "export * as ns from 'm';",
    'JsExportDefaultDeclaration' : 'export default 1;',
    'JsExportNamedDeclaration'   : 'var a, c; export { a as b, c };',
    'JsExportSpecifier'          : 'var a, c; export { a as b, c };',
    'JsExpressionStatement'      : 'a;',
    'JsForInStatement'           : 'for (var k in o) { a; }',
    'JsForOfStatement'           : 'for (var v of o) { a; }',
    'JsForStatement'             : 'for (var i = 0; i < 2; i++) { a; }',
    'JsFunctionDeclaration'      : 'function f(a, b) { return a; }',
    'JsFunctionExpression'       : 'var f = function (a, b) { return a; };',
    'JsIdentifier'               : 'a;',
    'JsIfStatement'              : 'if (a) { b; } else { c; }',
    'JsImportAttribute'          : "import a from 'm' with { type: 'json', x: 'y' };",
    'JsImportDeclaration'        : "import a, { b as c } from 'm';",
    'JsImportDefaultSpecifier'   : "import a, { b as c } from 'm';",
    'JsImportExpression'         : "import('m', { with: { type: 'json' } });",
    'JsImportNamespaceSpecifier' : "import * as ns from 'm';",
    'JsImportSpecifier'          : "import { a as b, c } from 'm';",
    'JsLabeledStatement'         : 'outer: while (a) { break outer; }',
    'JsLogicalExpression'        : 'a && b;',
    'JsMemberExpression'         : 'a.b;',
    'JsMetaProperty'             : 'import.meta;',
    'JsMethodDefinition'         : 'class C { m() {} static n() {} }',
    'JsNewExpression'            : 'new C(1, 2);',
    'JsNullLiteral'              : 'var x = null;',
    'JsNumericLiteral'           : 'var x = 1;',
    'JsObjectExpression'         : 'var o = { a: 1, b: 2 };',
    'JsObjectPattern'            : 'var { a, b } = o;',
    'JsParenthesizedExpression'  : '(a);',
    'JsPrivateIdentifier'        : 'class C { #x = 1; m() { return this.#x; } }',
    'JsProperty'                 : 'var o = { a: 1, b: 2 };',
    'JsPropertyDefinition'       : 'class C { x = 1; static y = 2; }',
    'JsRegExpLiteral'            : 'var r = /ab+c/gi;',
    'JsRestElement'              : 'function f(a, ...rest) { return rest; }',
    'JsReturnStatement'          : 'function f() { return 1; }',
    'JsScript'                   : 'a; b;',
    'JsSequenceExpression'       : 'a, b;',
    'JsSpreadElement'            : 'f(...a, ...b);',
    'JsStaticBlock'              : 'class C { static { a; b; } }',
    'JsStringLiteral'            : "var s = 'a';",
    'JsSwitchCase'               : 'switch (a) { case 1: b; break; default: c; }',
    'JsSwitchStatement'          : 'switch (a) { case 1: b; break; default: c; }',
    'JsTaggedTemplateExpression' : 'tag`a${b}c${d}e`;',
    'JsTemplateElement'          : '`a${b}c${d}e`;',
    'JsTemplateLiteral'          : '`a${b}c${d}e`;',
    'JsThisExpression'           : 'this;',
    'JsThrowStatement'           : 'throw a;',
    'JsTryStatement'             : 'try { a; } catch (e) { b; } finally { c; }',
    'JsUnaryExpression'          : 'typeof a;',
    'JsUpdateExpression'         : 'a++;',
    'JsVariableDeclaration'      : 'var a = 1, b = 2;',
    'JsVariableDeclarator'       : 'var a = 1, b = 2;',
    'JsWhileStatement'           : 'while (a) { b; }',
    'JsWithStatement'            : 'with (o) { a; }',
    'JsYieldExpression'          : 'function* g(h) { yield 1; yield* h(); }',
}

#: The shortest a child list may be and still spell something. A list absent from here may be empty,
#: which is most of them: a call with no arguments, an array with no elements and a class with no
#: members are all programs. The two that are not are the ones a pass reaches by removing the last
#: entry of a list it was editing.
SHORTEST_LIST: dict[str, dict[str, int]] = {
    'JsSequenceExpression'  : {'expressions': 2},
    'JsVariableDeclaration' : {'declarations': 1},
}


def shapes_the_grammar_forbids(root: Node) -> list[str]:
    """
    Every part of the tree at `root` that no JavaScript text spells, named one at a time.

    This is what `refinery.lib.scripts.Node.has_spelling` answers for a model that declares it, and
    the js model declares it only for a literal the source left unclosed. Until it says more, the
    rules live here, stated from the grammar rather than from the parser: a `var` declares at least
    one name, a comma expression joins at least two, a template's runs and holes alternate so there
    is exactly one more run than there are holes, and a name is at least one character.

    Two callers need the answer for opposite reasons. The generators use it to avoid manufacturing a
    tree that is not a program, since printing one says nothing about whether printing is faithful.
    The parser is held to it, because a shape with no text is one no parser may hand back: the
    synthesizer would have to invent a spelling for it, and inventing one is how a script quietly
    starts meaning something else.
    """
    found: list[str] = []
    for node in root.walk():
        name = type(node).__name__
        if getattr(node, 'name', None) == '':
            found.append(F'{name} has no name')
        if isinstance(node, JsTemplateLiteral):
            runs, holes = len(node.quasis), len(node.expressions)
            if runs != holes + 1:
                found.append(F'JsTemplateLiteral has {runs} runs around {holes} holes')
        for attribute, shortest in SHORTEST_LIST.get(name, {}).items():
            held = len(getattr(node, attribute))
            if held < shortest:
                found.append(F'{name} holds {held} of {attribute}, fewer than {shortest}')
    return found
