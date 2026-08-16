"""
Where strict mode comes from, and whether a deobfuscation leaves it where it was.

A `'use strict'` is the Use Strict Directive only by virtue of its position: it must be a string
literal standing in the run of string-literal statements that a Script, a function body or a class
static block opens with. A body that merely holds a statement list — a plain block, the body of a
`try`, a `catch` or a loop, a labelled statement, a `switch` case — reads no Directive Prologue at
all, and a `'use strict'` written at the head of one is an ordinary statement that computes a string
and discards it. Everything a strict body encloses is strict in turn, and every part of a class
definition is strict whatever encloses it.

Node decides all of it, through three probes and no reading of the specification.

The first is an octal literal. It is a number where the code around it is sloppy and a SyntaxError
where that code is strict, so writing one below a `'use strict'` asks the engine which mode it
compiled that position in: a program Node refuses is one where the directive was read as such, and a
program it reads is one where the directive was an ordinary statement. The probe is a fact about
where a piece of code stands, so it is the same question
`refinery.lib.scripts.js.strict.strict_mode_at` answers of the node standing there.

The second is a parameter list. A function whose parameters are anything but plain identifiers may
hold no Use Strict Directive at all, so `function f(a = 1) { 'use strict'; }` is a SyntaxError while
`function f(a) { 'use strict'; }` is a program. That is the one place the language says out loud
where a directive is permitted, and it is what
`refinery.lib.scripts.js.strict.has_simple_parameters` is asked for.

The third is the name `eval`, which strict code may not bind. An octal literal has to be written
where an expression goes, and the region a function body's directive governs reaches two places that
are not expression positions at all: the name the function binds and its parameter list. Node
refuses to read a program that binds `eval` in either, so writing the name there asks the same
question of a position no number can stand in.

A class static block is the one body no probe of the three reaches: class code is strict whatever
stands at the head of it, so the directive there decides nothing and both spellings of the program
are refused alike. It is recorded with that control beside it.

SECURITY: every snippet here is hand-authored and benign, and running it is what makes the engine
the oracle. Nothing from `samples` may ever be fed to this.
"""
from __future__ import annotations

import unittest

from typing import Sequence

from test import TestBase
from test.lib.scripts.js.analysis.differential import (
    JsEvaluation,
    completion_values,
    host_behavior,
    node_executable,
)

from refinery.lib.scripts import Node
from refinery.lib.scripts.js.model import (
    JsArrowFunctionExpression,
    JsExpressionStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsNumericLiteral,
    JsStringLiteral,
)
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.strict import (
    declares_use_strict,
    directive_prologue,
    has_simple_parameters,
    is_prologue_host,
    joins_directive_prologue,
    strict_mode_at,
)
from refinery.units.scripting.js import js

#: What `test.lib.scripts.js.analysis.differential.completion_values` reports for a text Node does
#: not read as a program at all.
NOT_A_PROGRAM = 'throw SyntaxError'

#: The Use Strict Directive as it is written into each program below, so that the same program
#: without it is the one this text is taken back out of.
THE_DIRECTIVE = "'use strict'; "

#: A program that writes the directive at the head of a body a Directive Prologue opens, with an
#: octal literal below it. Node refuses every one of them, and reads every one of them once the
#: directive is removed.
A_BODY_A_PROLOGUE_OPENS = {
    'a script'               : "'use strict'; 010;",
    'a function declaration' : "function f() { 'use strict'; 010; }",
    'a function expression'  : "var f = function () { 'use strict'; 010; };",
    'an arrow with a block'  : "var f = () => { 'use strict'; 010; };",
    'a method'               : "var o = { m() { 'use strict'; 010; } };",
    'a getter'               : "var o = { get g() { 'use strict'; 010; } };",
    'a setter'               : "var o = { set s(v) { 'use strict'; 010; } };",
    'a generator'            : "function* g() { 'use strict'; 010; }",
    'an async function'      : "async function h() { 'use strict'; 010; }",
    'a nested function'      : "function f() { function g() { 'use strict'; 010; } }",
}

#: A program that writes the directive at the head of a statement list no Directive Prologue opens.
#: Node reads every one of them, with and without the directive: the statement is a string that is
#: computed and discarded, and the octal literal below it stands in sloppy code.
A_BODY_NO_PROLOGUE_OPENS = {
    'a plain block'          : "{ 'use strict'; 010; }",
    'an if branch'           : "if (1) { 'use strict'; 010; }",
    'an else branch'         : "if (0) ; else { 'use strict'; 010; }",
    'a try block'            : "try { 'use strict'; 010; } catch (e) {}",
    'a catch block'          : "try {} catch (e) { 'use strict'; 010; }",
    'a finally block'        : "try {} finally { 'use strict'; 010; }",
    'a for body'             : "for (;;) { 'use strict'; 010; break; }",
    'a for-in body'          : "for (var k in {}) { 'use strict'; 010; }",
    'a for-of body'          : "for (var v of []) { 'use strict'; 010; }",
    'a while body'           : "while (0) { 'use strict'; 010; }",
    'a do-while body'        : "do { 'use strict'; 010; } while (0);",
    'a switch case'          : "switch (1) { case 1: 'use strict'; 010; }",
    'a switch default'       : "switch (1) { default: 'use strict'; 010; }",
    'a labelled block'       : "L: { 'use strict'; 010; }",
    'a with block'           : "with ({}) { 'use strict'; 010; }",
    'a block inside a body'  : "function f() { { 'use strict'; 010; } }",
}

#: The one body whose directive neither probe reaches. A class static block opens a Directive
#: Prologue, and every part of a class definition is strict already, so Node refuses this program
#: and refuses it just the same with the directive taken back out.
A_CLASS_STATIC_BLOCK = "class C { static { 'use strict'; 010; } }"

#: A position an octal literal is written at, mapped to whether the code there is strict. Nothing
#: below declares a mode: each of these is a class, whose every part is strict however sloppy the
#: file around it is, or a body nested inside one that declares it.
AN_OCTAL_LITERAL_STANDING_IN = {
    'a sloppy script'                    : ('010;', False),
    'a sloppy function body'             : ('function f() { 010; }', False),
    'a strict script'                    : ("'use strict'; 010;", True),
    'a function inside a strict script'  : ("'use strict'; function f() { 010; }", True),
    'an arrow inside a strict script'    : ("'use strict'; var f = () => 010;", True),
    'a block inside a strict script'     : ("'use strict'; { 010; }", True),
    'a function inside a strict body'    : (
        "function f() { 'use strict'; function g() { 010; } }", True),
    'a catch inside a strict body'       : (
        "function f() { 'use strict'; try {} catch (e) { 010; } }", True),
    'a class method'                     : ('class C { m() { 010; } }', True),
    'a class field initializer'          : ('class C { p = 010; }', True),
    'a class static block'               : ('class C { static { 010; } }', True),
    'a class heritage clause'            : ('class C extends (010, Object) {}', True),
    'a class computed key'               : ('class C { [010]() {} }', True),
    'a function inside a class method'   : ('class C { m() { function g() { 010; } } }', True),
    'a class expression method'          : ('var C = class { m() { 010; } };', True),
    'a script below a strict function'   : ("function f() { 'use strict'; } 010;", False),
    'a script below a block holding one' : ("{ 'use strict'; } 010;", False),
    'a body below a block holding one'   : ("function f() { { 'use strict'; } 010; }", False),
}

#: A program that binds the name `eval` inside the code a function body's Use Strict Directive
#: governs. Strict code may bind neither `eval` nor `arguments`, so Node refuses every one of these,
#: and reads every one of them once the directive is taken back out.
A_BINDING_THE_DIRECTIVE_GOVERNS = {
    'the name of a declaration'      : "function eval() { 'use strict'; }",
    'the name of an expression'      : "var f = function eval() { 'use strict'; };",
    'the name of a generator'        : "function* eval() { 'use strict'; }",
    'the name of an async function'  : "async function eval() { 'use strict'; }",
    'the name of a nested function'  : "function f() { function eval() { 'use strict'; } }",
    'the parameter of a declaration' : "function f(eval) { 'use strict'; }",
    'a later parameter'              : "function f(a, eval) { 'use strict'; }",
    'the parameter of an expression' : "var f = function (eval) { 'use strict'; };",
    'the parameter of a named one'   : "var f = function g(eval) { 'use strict'; };",
    'the parameter of an arrow'      : "var f = (eval) => { 'use strict'; };",
    'the parameter of a method'      : "var o = { m(eval) { 'use strict'; } };",
    'the parameter of a setter'      : "var o = { set s(eval) { 'use strict'; } };",
    'the parameter of a generator'   : "function* g(eval) { 'use strict'; }",
    'the parameter of an async one'  : "async function h(eval) { 'use strict'; }",
    'a binding inside the body'      : "function f() { 'use strict'; var eval = 1; }",
}

#: A program that binds the same name outside that code, which Node reads. A directive governs the
#: one function it was written in and nothing around it, and two of these bind the name in a place
#: that only looks as though it belongs to that function: a property key is a name and not a
#: binding, and the variable a function expression is stored in belongs to the statement that
#: declares it.
A_BINDING_OUTSIDE_WHAT_IT_GOVERNS = {
    'a binding above the function'       : "var eval = 1; function f() { 'use strict'; }",
    'a binding below the function'       : "function f() { 'use strict'; } var eval = 1;",
    'the name of a sibling function'     : "function eval() {} function f() { 'use strict'; }",
    'the key of a method'                : "var o = { eval() { 'use strict'; } };",
    'the key of a getter'                : "var o = { get eval() { 'use strict'; } };",
    'the variable an arrow is put in'    : "var eval = () => { 'use strict'; };",
    'the variable a function is put in'  : "var eval = function () { 'use strict'; };",
    'the name of the function around'    : "function eval() { function g() { 'use strict'; } }",
    'a parameter of the function around' : "function f(eval) { function g() { 'use strict'; } }",
}

#: A statement standing between the head of a body and a `'use strict'` written below it, mapped to
#: whether the prologue still reaches that directive. The empty key is the directive standing first.
A_STATEMENT_ABOVE_THE_DIRECTIVE = {
    ''                    : True,
    "'other';"            : True,
    '"another";'          : True,
    "'a'; 'b';"           : True,
    R"'use\u0020strict';" : True,
    ';'                   : False,
    '0;'                  : False,
    'var x;'              : False,
    "('paren');"          : False,
    "'a' + 'b';"          : False,
    "'a', 'b';"           : False,
    'x = 1;'              : False,
    "{ 'x'; }"            : False,
    "if (0) 'x';"         : False,
    'function g() {}'     : False,
    "'a'.length;"         : False,
}

#: The statements a body is written with. Its Directive Prologue is the run of string literals it
#: opens with, which is the first two, and a directive inserted at any index from zero up to and
#: including two is read as one.
A_BODY_WRITTEN_AS = ["'alpha';", "'beta';", '0;', "'gamma';"]

#: A statement that is no string literal and that a fold would write as one, placed at a position,
#: mapped to whether respelling it hands a `'use strict'` below it the Directive Prologue.
A_STATEMENT_STANDING_AT = {
    'the head of a script'           : ("{stmt} 'use strict'; 010;", True),
    'the second statement'           : ("'lead'; {stmt} 'use strict'; 010;", True),
    'below a statement that is none' : ("0; {stmt} 'use strict'; 010;", False),
    'the head of a function body'    : ("function f() {{ {stmt} 'use strict'; 010; }}", True),
    'the head of a plain block'      : ("{{ {stmt} 'use strict'; 010; }}", False),
    'the head of a switch case'      : (
        "switch (1) {{ case 1: {stmt} 'use strict'; 010; }}", False),
    'the head of a catch block'      : (
        "try {{}} catch (e) {{ {stmt} 'use strict'; 010; }}", False),
    'the head of a nested block'     : (
        "function f() {{ {{ {stmt} 'use strict'; 010; }} }}", False),
}

#: The statement `A_STATEMENT_STANDING_AT` places. It is no string literal, so it ends whatever
#: Directive Prologue it stands in, and it denotes a string, so a fold writes one where it stood.
A_COMPUTED_STRING = "'a' + 'b';"

#: The string literal that fold writes. It does not denote `use strict`, so it is never the
#: directive itself; what it can do is hand the position of one to a statement below it.
A_PLAIN_STRING = "'ab';"

#: A parameter list, mapped to whether it is a simple one. Every parameter of a simple list is a
#: plain identifier; a default, a rest element and a destructuring pattern are each something else.
#: The empty list is simple, nothing in it being anything but an identifier.
A_PARAMETER_LIST = {
    ''          : True,
    'a'         : True,
    'a, b'      : True,
    'a, b, c'   : True,
    'a,'        : True,
    'a = 1'     : False,
    'a, b = 2'  : False,
    '...a'      : False,
    'a, ...b'   : False,
    '[a]'       : False,
    '[a, b]'    : False,
    '{a}'       : False,
    '{a: b}'    : False,
    '{a} = {}'  : False,
    'a, [b]'    : False,
}

#: Each way of writing a function that takes a parameter list and holds a body.
A_FUNCTION_WRITTEN_AS = {
    'a function declaration' : 'function f({params}) {{ {body} }}',
    'a function expression'  : 'var f = function ({params}) {{ {body} }};',
    'an arrow'               : 'var f = ({params}) => {{ {body} }};',
    'a method'               : 'var o = {{ m({params}) {{ {body} }} }};',
    'a generator'            : 'function* g({params}) {{ {body} }}',
    'an async function'      : 'async function h({params}) {{ {body} }}',
}

#: A statement that prints `true` where it stands in strict code and `false` where it stands in
#: sloppy code. A plain call passes no receiver, so `this` in the callee is `undefined` under strict
#: and the global object under sloppy, and a function written inside a body runs in that body's
#: mode.
_REPORTS_THE_MODE_IT_STANDS_IN = (
    'console.log((function () { return this; })() === undefined);'
)

#: A program that says which mode each part of it runs in, mapped to what Node prints for it.
A_PROGRAM_REPORTING_ITS_MODE = {
    F"'use strict';\n{_REPORTS_THE_MODE_IT_STANDS_IN}":
        'true\n',
    F"'use strict';\nfunction f() {{ {_REPORTS_THE_MODE_IT_STANDS_IN} }}\nf();":
        'true\n',
    F"function f() {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }}\nf();":
        'true\n',
    F"var o = {{ m() {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }} }};\no.m();":
        'true\n',
    F"var f = () => {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }};\nf();":
        'true\n',
    F"function* g() {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }}\ng().next();":
        'true\n',
    F"var o = {{ get p() {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }} }};\no.p;":
        'true\n',
    F'class C {{ m() {{ {_REPORTS_THE_MODE_IT_STANDS_IN} }} }}\nnew C().m();':
        'true\n',
    F'class C {{ static {{ {_REPORTS_THE_MODE_IT_STANDS_IN} }} }}':
        'true\n',
    F'class C {{ p = (function () {{ {_REPORTS_THE_MODE_IT_STANDS_IN} }})(); }}\nnew C();':
        'true\n',
    F"function f() {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }}\n"
    F'function g() {{ {_REPORTS_THE_MODE_IT_STANDS_IN} }}\nf();\ng();':
        'true\nfalse\n',
    F"{_REPORTS_THE_MODE_IT_STANDS_IN}\n"
    F"(function () {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }})();":
        'false\ntrue\n',
    F"switch (1) {{ case 1: 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }}":
        'false\n',
    F"try {{ throw 1; }} catch (e) {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }}":
        'false\n',
    F"for (var i = 0; i < 1; i++) {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }}":
        'false\n',
    F"L: {{ 'use strict'; {_REPORTS_THE_MODE_IT_STANDS_IN} }}":
        'false\n',
    F"console.log('x');\n'use strict';\n{_REPORTS_THE_MODE_IT_STANDS_IN}":
        'x\nfalse\n',
    F"'note';\n'use strict';\n{_REPORTS_THE_MODE_IT_STANDS_IN}":
        'true\n',
    "function f(a) { 'use strict'; a = 2; return arguments[0]; }\nconsole.log(f(1));":
        '1\n',
    "'use strict';\nvar s = ['con', 'sole'];\nconsole.log(s[0] + s[1]);\n"
    + _REPORTS_THE_MODE_IT_STANDS_IN:
        'console\ntrue\n',
}

#: A read a constant string decides, standing at the head of a body that reads no Directive
#: Prologue, mapped to the text `refinery.js` writes for the whole program. The read answers a
#: string, so folding it writes a string literal where the read stood; no directive is ever read
#: there, so no mode turns on the rewrite and nothing licenses declining it.
A_READ_WHERE_NO_PROLOGUE_IS_READ = {
    "{ 'abc'[0]; }":
        "{\n  'a';\n}",
    "switch (1) { case 1: 'abc'[0]; }":
        "switch (1) {\n  case 1:\n    'a';\n}",
    "try {} catch (e) { 'abc'[0]; }":
        "try {} catch (e) {\n  'a';\n}",
    "L: { 'abc'[0]; }":
        "L: {\n  'a';\n}",
    "function f() { { 'abc'[0]; } }":
        "function f() {\n  {\n    'a';\n  }\n}",
}

#: The same read standing where a string literal would join a Directive Prologue, mapped to the text
#: `refinery.js` writes for it. Folding it there hands the prologue a statement that had ended one.
A_READ_WHERE_A_PROLOGUE_IS_READ = {
    "'abc'[0];":
        "'abc'[0];",
    "function f() { 'abc'[0]; }":
        "function f() {\n  'abc'[0];\n}",
}


def _refused(programs: Sequence[str]) -> list[bool]:
    """
    Whether Node refuses to read each of *programs* as a program at all. Each is compiled and run in
    a context of its own, so a refusal is a fact about that one text.
    """
    return [
        value == NOT_A_PROGRAM
        for value in completion_values(programs, JsEvaluation.SCRIPT)
    ]


def _without_the_directive(source: str) -> str:
    return source.replace(THE_DIRECTIVE, '', 1)


def _the_directive_statement(root: Node) -> JsExpressionStatement:
    """
    The statement of *root* that is written as the Use Strict Directive.
    """
    statement, = (
        node for node in root.walk()
        if isinstance(node, JsExpressionStatement)
        and isinstance(node.expression, JsStringLiteral)
        and node.expression.body == 'use strict'
    )
    return statement


def _the_binding_named_eval(root: Node) -> JsIdentifier:
    name, = (
        node for node in root.walk()
        if isinstance(node, JsIdentifier) and node.name == 'eval'
    )
    return name


def _the_octal_literal(root: Node) -> JsNumericLiteral:
    literal, = (
        node for node in root.walk()
        if isinstance(node, JsNumericLiteral) and node.raw == '010'
    )
    return literal


def _the_computed_string_statement(root: Node) -> JsExpressionStatement:
    """
    The statement `A_STATEMENT_STANDING_AT` places, found through the literal it is written with
    rather than through its position, which is what each row varies.
    """
    statement, = (
        node for node in root.walk()
        if isinstance(node, JsExpressionStatement)
        and any(
            isinstance(inner, JsStringLiteral) and inner.body == 'a'
            for inner in node.walk()
        )
    )
    return statement


def _the_function(
    root: Node,
) -> JsFunctionDeclaration | JsFunctionExpression | JsArrowFunctionExpression:
    function, = (
        node for node in root.walk()
        if isinstance(
            node,
            (JsFunctionDeclaration, JsFunctionExpression, JsArrowFunctionExpression),
        )
    )
    return function


def _prologue_spellings(host: Node) -> list[str]:
    """
    What each statement of the Directive Prologue of *host* is spelled with.
    """
    return [
        statement.expression.body
        for statement in directive_prologue(host)
        if isinstance(statement.expression, JsStringLiteral)
    ]


def _a_script_whose_directive_stands_below(head: str) -> str:
    return F"{head} 'use strict'; 010;"


def _a_function_body_whose_directive_stands_below(head: str) -> str:
    return F"function f() {{ {head} 'use strict'; 010; }}"


def _a_body_with_the_directive_at(index: int) -> str:
    statements = list(A_BODY_WRITTEN_AS)
    statements.insert(index, "'use strict';")
    return F'{" ".join(statements)} 010;'


def _deobfuscated(source: str) -> str:
    return source.encode('utf8') | js() | str


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestWhichBodiesReadADirectivePrologue(TestBase):

    def test_a_directive_at_the_head_of_a_prologue_host_makes_the_code_below_it_strict(self):
        sources = list(A_BODY_A_PROLOGUE_OPENS.values())
        self.assertEqual(_refused(sources), [True] * len(sources))

    def test_the_same_body_without_the_directive_is_sloppy(self):
        sources = [_without_the_directive(s) for s in A_BODY_A_PROLOGUE_OPENS.values()]
        self.assertEqual(_refused(sources), [False] * len(sources))

    def test_a_directive_at_the_head_of_any_other_statement_list_changes_no_mode(self):
        sources = list(A_BODY_NO_PROLOGUE_OPENS.values())
        self.assertEqual(_refused(sources), [False] * len(sources))

    def test_those_bodies_are_sloppy_without_the_directive_too(self):
        sources = [_without_the_directive(s) for s in A_BODY_NO_PROLOGUE_OPENS.values()]
        self.assertEqual(_refused(sources), [False] * len(sources))

    def test_a_class_static_block_is_strict_whatever_stands_at_its_head(self):
        self.assertEqual(
            _refused([A_CLASS_STATIC_BLOCK, _without_the_directive(A_CLASS_STATIC_BLOCK)]),
            [True, True],
        )


class TestIsPrologueHostAnswersForTheBodiesTheEngineReadsOneIn(TestBase):

    def _hosts(self, sources: Sequence[str]) -> list[bool]:
        return [
            is_prologue_host(_the_directive_statement(JsParser(source).parse()).parent)
            for source in sources
        ]

    def test_a_body_the_engine_compiled_strict_holds_a_prologue(self):
        sources = list(A_BODY_A_PROLOGUE_OPENS.values()) + [A_CLASS_STATIC_BLOCK]
        self.assertEqual(self._hosts(sources), [True] * len(sources))

    def test_a_body_the_engine_left_sloppy_holds_none(self):
        sources = list(A_BODY_NO_PROLOGUE_OPENS.values())
        self.assertEqual(self._hosts(sources), [False] * len(sources))


class TestStrictModeAtAnswersTheModeTheEngineCompiled(TestBase):

    def _modes(self, sources: Sequence[str]) -> list[bool]:
        return [
            strict_mode_at(_the_octal_literal(JsParser(source).parse()))
            for source in sources
        ]

    def test_the_octal_literal_below_a_directive_a_body_reads_stands_in_strict_code(self):
        sources = list(A_BODY_A_PROLOGUE_OPENS.values()) + [A_CLASS_STATIC_BLOCK]
        self.assertEqual(self._modes(sources), [True] * len(sources))

    def test_the_octal_literal_below_a_directive_no_body_reads_stands_in_sloppy_code(self):
        sources = list(A_BODY_NO_PROLOGUE_OPENS.values())
        self.assertEqual(self._modes(sources), [False] * len(sources))

    def test_a_position_a_strict_body_encloses_inherits_the_mode_of_that_body(self):
        rows = AN_OCTAL_LITERAL_STANDING_IN
        self.assertEqual(
            self._modes([source for source, _ in rows.values()]),
            [strict for _, strict in rows.values()],
        )

    def _eval_modes(self, sources: Sequence[str]) -> list[bool]:
        return [
            strict_mode_at(_the_binding_named_eval(JsParser(source).parse()))
            for source in sources
        ]

    def test_the_name_and_the_parameters_of_a_function_stand_in_the_mode_of_its_body(self):
        sources = list(A_BINDING_THE_DIRECTIVE_GOVERNS.values())
        self.assertEqual(self._eval_modes(sources), [True] * len(sources))

    def test_a_binding_the_directive_does_not_govern_stands_in_sloppy_code(self):
        sources = list(A_BINDING_OUTSIDE_WHAT_IT_GOVERNS.values())
        self.assertEqual(self._eval_modes(sources), [False] * len(sources))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestModeIsInheritedByEverythingAStrictBodyEncloses(TestBase):

    def test_node_compiles_each_position_in_the_mode_the_corpus_records(self):
        rows = AN_OCTAL_LITERAL_STANDING_IN
        self.assertEqual(
            _refused([source for source, _ in rows.values()]),
            [strict for _, strict in rows.values()],
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestWhatAFunctionBodysDirectiveGovernsBesidesThatBody(TestBase):
    """
    A function body's directive governs the function and not the body alone. The name the function
    binds and its parameter list stand outside the braces the directive opens and are strict code
    all the same, which is what makes `function eval() { 'use strict'; }` and
    `function f(eval) { 'use strict'; }` two texts Node will not read.

    It reaches no further than that one function. The code around it stays sloppy, and so does
    anything that only looks as though it belongs to the function: the property key a method is
    written under is a name and not a binding, and the variable a function expression is assigned to
    belongs to the statement that declares it.
    """

    def test_node_refuses_the_name_bound_in_the_code_the_directive_governs(self):
        sources = list(A_BINDING_THE_DIRECTIVE_GOVERNS.values())
        self.assertEqual(_refused(sources), [True] * len(sources))

    def test_node_reads_the_same_name_bound_anywhere_else(self):
        sources = list(A_BINDING_OUTSIDE_WHAT_IT_GOVERNS.values())
        self.assertEqual(_refused(sources), [False] * len(sources))

    def test_it_is_the_directive_and_not_the_name_that_costs_the_program(self):
        sources = [
            _without_the_directive(source)
            for source in A_BINDING_THE_DIRECTIVE_GOVERNS.values()
        ]
        self.assertEqual(_refused(sources), [False] * len(sources))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestHowFarADirectivePrologueReaches(TestBase):

    def test_a_directive_is_read_below_string_literal_statements_and_below_nothing_else(self):
        rows = A_STATEMENT_ABOVE_THE_DIRECTIVE
        self.assertEqual(
            _refused([_a_script_whose_directive_stands_below(head) for head in rows]),
            list(rows.values()),
        )

    def test_a_function_body_reads_its_prologue_the_same_way_a_script_does(self):
        rows = A_STATEMENT_ABOVE_THE_DIRECTIVE
        self.assertEqual(
            _refused([_a_function_body_whose_directive_stands_below(head) for head in rows]),
            list(rows.values()),
        )

    def test_a_directive_stands_anywhere_from_the_head_up_to_the_end_of_the_prologue(self):
        indices = range(len(A_BODY_WRITTEN_AS) + 1)
        self.assertEqual(
            _refused([_a_body_with_the_directive_at(index) for index in indices]),
            [True, True, True, False, False],
        )


class TestTheDirectivePrologueTheToolReads(TestBase):

    def test_declares_use_strict_answers_what_the_engine_compiled(self):
        rows = A_STATEMENT_ABOVE_THE_DIRECTIVE
        self.assertEqual(
            [
                declares_use_strict(JsParser(_a_script_whose_directive_stands_below(head)).parse())
                for head in rows
            ],
            list(rows.values()),
        )

    def test_the_prologue_is_the_run_of_string_literal_statements_a_body_opens_with(self):
        script = JsParser(F'{" ".join(A_BODY_WRITTEN_AS)} 010;').parse()
        self.assertEqual(_prologue_spellings(script), ['alpha', 'beta'])

    def test_a_body_that_opens_with_no_string_literal_has_an_empty_prologue(self):
        script = JsParser("0; 'alpha'; 'beta';").parse()
        self.assertEqual(_prologue_spellings(script), [])


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestWhichStatementsWouldJoinADirectivePrologue(TestBase):

    def test_a_statement_that_is_no_string_literal_hands_no_directive_position_away(self):
        sources = [
            template.format(stmt=A_COMPUTED_STRING)
            for template, _ in A_STATEMENT_STANDING_AT.values()
        ]
        self.assertEqual(_refused(sources), [False] * len(sources))

    def test_respelling_it_as_a_string_literal_extends_the_prologue_over_what_follows(self):
        rows = A_STATEMENT_STANDING_AT
        self.assertEqual(
            _refused([template.format(stmt=A_PLAIN_STRING) for template, _ in rows.values()]),
            [joins for _, joins in rows.values()],
        )

    def test_joins_directive_prologue_answers_what_the_respelling_costs(self):
        rows = A_STATEMENT_STANDING_AT
        self.assertEqual(
            [
                joins_directive_prologue(
                    _the_computed_string_statement(
                        JsParser(template.format(stmt=A_COMPUTED_STRING)).parse()))
                for template, _ in rows.values()
            ],
            [joins for _, joins in rows.values()],
        )


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestWhichParameterListsADirectiveMayStandUnder(TestBase):

    def test_node_refuses_a_directive_under_every_list_that_is_not_simple(self):
        for shape, template in A_FUNCTION_WRITTEN_AS.items():
            sources = [
                template.format(params=params, body="'use strict';")
                for params in A_PARAMETER_LIST
            ]
            with self.subTest(shape=shape):
                self.assertEqual(
                    _refused(sources),
                    [not simple for simple in A_PARAMETER_LIST.values()],
                )

    def test_node_reads_every_one_of_those_lists_with_an_empty_body(self):
        for shape, template in A_FUNCTION_WRITTEN_AS.items():
            sources = [
                template.format(params=params, body='')
                for params in A_PARAMETER_LIST
            ]
            with self.subTest(shape=shape):
                self.assertEqual(_refused(sources), [False] * len(sources))


class TestHasSimpleParametersAnswersWhereADirectiveIsPermitted(TestBase):

    def test_a_list_is_simple_exactly_where_the_engine_permits_a_directive_under_it(self):
        for shape, template in A_FUNCTION_WRITTEN_AS.items():
            answers = [
                has_simple_parameters(
                    _the_function(JsParser(template.format(params=params, body='')).parse()))
                for params in A_PARAMETER_LIST
            ]
            with self.subTest(shape=shape):
                self.assertEqual(answers, list(A_PARAMETER_LIST.values()))


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestDeobfuscationLeavesEveryModeWhereItWas(TestBase):

    def test_node_prints_what_the_corpus_records_for_each_program(self):
        rows = A_PROGRAM_REPORTING_ITS_MODE
        self.assertEqual(
            {source: host_behavior(source) for source in rows},
            {source: (printed, None) for source, printed in rows.items()},
        )

    def test_the_deobfuscation_of_each_program_prints_the_same(self):
        rows = A_PROGRAM_REPORTING_ITS_MODE
        self.assertEqual(
            {source: host_behavior(_deobfuscated(source)) for source in rows},
            {source: (printed, None) for source, printed in rows.items()},
        )


class TestAFoldDeclinesOnlyWhereADirectiveCouldBeRead(TestBase):

    def test_a_read_in_a_body_that_opens_no_prologue_is_folded(self):
        rows = A_READ_WHERE_NO_PROLOGUE_IS_READ
        self.assertEqual({s: _deobfuscated(s) for s in rows}, dict(rows))

    def test_a_read_that_would_join_a_prologue_is_left_standing(self):
        rows = A_READ_WHERE_A_PROLOGUE_IS_READ
        self.assertEqual({s: _deobfuscated(s) for s in rows}, dict(rows))
