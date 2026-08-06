from __future__ import annotations

from test import TestBase

import itertools
import unittest

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.parser import Ps1Parser
from refinery.lib.scripts.ps1.model import (
    Ps1AccessKind,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1BinaryExpression,
    Ps1CastExpression,
    Ps1CommandInvocation,
    Ps1ScopeModifier,
    Ps1ExpandableString,
    Ps1HashLiteral,
    Ps1HereString,
    Ps1IndexExpression,
    Ps1IntegerLiteral,
    Ps1InvokeMember,
    Ps1MemberAccess,
    Ps1ParenExpression,
    Ps1RangeExpression,
    Ps1RealLiteral,
    Ps1ScriptBlock,
    Ps1StringLiteral,
    Ps1SubExpression,
    Ps1TypeExpression,
    Ps1UnaryExpression,
    Ps1Variable,
    Ps1ExpressionStatement,
    Ps1Script,
    Ps1ArrayExpression,
)


class TestPs1ParserExpressions(TestBase):

    def _parse_expr(self, source: str):
        p = Ps1Parser(source)
        script = p.parse()
        self.assertIsInstance(script, Ps1Script)
        self.assertTrue(len(script.body) > 0)
        stmt = script.body[0]
        self.assertIsInstance(stmt, Ps1ExpressionStatement)
        return stmt.expression

    def test_integer_literal(self):
        expr = self._parse_expr('42')
        self.assertIsInstance(expr, Ps1IntegerLiteral)
        self.assertEqual(expr.value, 42)

    def test_hex_literal(self):
        expr = self._parse_expr('0xDEAD')
        self.assertIsInstance(expr, Ps1IntegerLiteral)
        self.assertEqual(expr.value, 0xDEAD)

    def test_real_literal(self):
        expr = self._parse_expr('3.14')
        self.assertIsInstance(expr, Ps1RealLiteral)
        self.assertAlmostEqual(expr.value, 3.14)

    def test_string_literal_verbatim(self):
        expr = self._parse_expr("'hello'")
        self.assertIsInstance(expr, Ps1StringLiteral)
        self.assertEqual(expr.value, 'hello')

    def test_string_literal_expandable_no_vars(self):
        for oq, cq in itertools.chain(
            itertools.product("""'‘’‚‛""", repeat=2),
            itertools.product('''"“”„''', repeat=2),
        ):
            expr = self._parse_expr(F'{oq}hello{cq}')
            self.assertIsInstance(expr, Ps1StringLiteral)
            self.assertEqual(expr.value, 'hello')

    def test_expandable_string_with_variable(self):
        expr = self._parse_expr('"hello $name"')
        self.assertIsInstance(expr, Ps1ExpandableString)
        self.assertTrue(len(expr.parts) >= 2)

    def test_expandable_string_with_subexpression(self):
        expr = self._parse_expr('"result: $(1+2)"')
        self.assertIsInstance(expr, Ps1ExpandableString)
        has_subexpr = any(isinstance(p, Ps1SubExpression) for p in expr.parts)
        self.assertTrue(has_subexpr)

    def test_variable(self):
        expr = self._parse_expr('$x')
        self.assertIsInstance(expr, Ps1Variable)
        self.assertEqual(expr.name, 'x')

    def test_variable_scoped(self):
        expr = self._parse_expr('$env:PATH')
        self.assertIsInstance(expr, Ps1Variable)
        self.assertEqual(expr.name, 'PATH')

    def test_variable_drive_qualified(self):
        expr = self._parse_expr('$HKLM:Software')
        self.assertIsInstance(expr, Ps1Variable)
        self.assertEqual(expr.scope, Ps1ScopeModifier.DRIVE)
        self.assertEqual(expr.name, 'Software')

    def test_addition(self):
        expr = self._parse_expr('1 + 2')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '+')
        self.assertIsInstance(expr.left, Ps1IntegerLiteral)
        self.assertIsInstance(expr.right, Ps1IntegerLiteral)

    def test_precedence_mul_over_add(self):
        expr = self._parse_expr('1 + 2 * 3')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '+')
        self.assertIsInstance(expr.right, Ps1BinaryExpression)
        self.assertEqual(expr.right.operator, '*')

    def test_comparison(self):
        expr = self._parse_expr('$x -eq 1')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '-eq')

    def test_logical_operators(self):
        expr = self._parse_expr('$a -and $b -or $c')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '-or')
        self.assertIsInstance(expr.left, Ps1BinaryExpression)
        self.assertEqual(expr.left.operator, '-and')

    def test_bitwise_operators(self):
        expr = self._parse_expr('$a -band 0xFF')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '-band')

    def test_range_expression(self):
        expr = self._parse_expr('1..10')
        self.assertIsInstance(expr, Ps1RangeExpression)
        self.assertIsInstance(expr.start, Ps1IntegerLiteral)
        self.assertIsInstance(expr.end, Ps1IntegerLiteral)

    def test_array_literal(self):
        expr = self._parse_expr('1, 2, 3')
        self.assertIsInstance(expr, Ps1ArrayLiteral)
        self.assertEqual(len(expr.elements), 3)

    def test_unary_not(self):
        expr = self._parse_expr('-not $x')
        self.assertIsInstance(expr, Ps1UnaryExpression)
        self.assertEqual(expr.operator, '-not')
        self.assertTrue(expr.prefix)

    def test_unary_negation(self):
        expr = self._parse_expr('-5')
        self.assertIsInstance(expr, Ps1UnaryExpression)
        self.assertEqual(expr.operator, '-')

    def test_unary_exclaim(self):
        expr = self._parse_expr('!$x')
        self.assertIsInstance(expr, Ps1UnaryExpression)
        self.assertEqual(expr.operator, '!')

    def test_prefix_increment(self):
        expr = self._parse_expr('++$x')
        self.assertIsInstance(expr, Ps1UnaryExpression)
        self.assertEqual(expr.operator, '++')
        self.assertTrue(expr.prefix)

    def test_postfix_increment(self):
        expr = self._parse_expr('$x++')
        self.assertIsInstance(expr, Ps1UnaryExpression)
        self.assertEqual(expr.operator, '++')
        self.assertFalse(expr.prefix)

    def test_unary_csplit(self):
        expr = self._parse_expr('-csplit "hello world"')
        self.assertIsInstance(expr, Ps1UnaryExpression)
        self.assertEqual(expr.operator, '-csplit')
        self.assertTrue(expr.prefix)

    def test_unary_isplit(self):
        expr = self._parse_expr('-isplit "hello world"')
        self.assertIsInstance(expr, Ps1UnaryExpression)
        self.assertEqual(expr.operator, '-isplit')
        self.assertTrue(expr.prefix)

    def test_cast_expression(self):
        expr = self._parse_expr('[int]$x')
        self.assertIsInstance(expr, Ps1CastExpression)
        self.assertEqual(expr.type_name, 'int')

    def test_type_expression(self):
        expr = self._parse_expr('[string]')
        self.assertIsInstance(expr, Ps1TypeExpression)
        self.assertEqual(expr.name, 'string')

    def test_member_access(self):
        expr = self._parse_expr('$s.Length')
        self.assertIsInstance(expr, Ps1MemberAccess)
        self.assertEqual(expr.member, 'Length')

    def test_static_member_access(self):
        expr = self._parse_expr('[int]::MaxValue')
        self.assertIsInstance(expr, Ps1MemberAccess)
        self.assertEqual(expr.member, 'MaxValue')

    def test_method_invocation(self):
        expr = self._parse_expr('$s.Substring(0, 5)')
        self.assertIsInstance(expr, Ps1InvokeMember)
        self.assertEqual(expr.member, 'Substring')
        self.assertEqual(len(expr.arguments), 2)

    def test_static_method_invocation(self):
        expr = self._parse_expr('[System.Text.Encoding]::UTF8.GetBytes("test")')
        self.assertIsInstance(expr, Ps1InvokeMember)

    def test_index_expression(self):
        expr = self._parse_expr('$arr[0]')
        self.assertIsInstance(expr, Ps1IndexExpression)
        self.assertIsInstance(expr.index, Ps1IntegerLiteral)

    def test_paren_expression(self):
        expr = self._parse_expr('(1 + 2)')
        self.assertIsInstance(expr, Ps1ParenExpression)

    def test_sub_expression(self):
        expr = self._parse_expr('$($x + $y)')
        self.assertIsInstance(expr, Ps1SubExpression)

    def test_array_expression(self):
        expr = self._parse_expr('@(1, 2, 3)')
        self.assertIsInstance(expr, Ps1ArrayExpression)

    def test_hash_literal(self):
        expr = self._parse_expr('@{ a = 1; b = 2 }')
        self.assertIsInstance(expr, Ps1HashLiteral)
        self.assertEqual(len(expr.pairs), 2)

    def test_hash_literal_paren_key(self):
        expr = self._parse_expr('@{ (1+2) = "three" }')
        self.assertIsInstance(expr, Ps1HashLiteral)
        self.assertEqual(len(expr.pairs), 1)
        self.assertIsInstance(expr.pairs[0][0], Ps1ParenExpression)

    def test_hash_literal_negative_integer_key(self):
        expr = self._parse_expr('@{ -1 = "neg" }')
        self.assertIsInstance(expr, Ps1HashLiteral)
        self.assertEqual(len(expr.pairs), 1)
        self.assertIsInstance(expr.pairs[0][0], Ps1UnaryExpression)

    def test_hash_literal_real_key(self):
        expr = self._parse_expr('@{ 3.14 = "pi" }')
        self.assertIsInstance(expr, Ps1HashLiteral)
        self.assertEqual(len(expr.pairs), 1)
        self.assertIsInstance(expr.pairs[0][0], Ps1RealLiteral)

    def test_hash_literal_subexpression_key(self):
        expr = self._parse_expr('@{ $("key") = "val" }')
        self.assertIsInstance(expr, Ps1HashLiteral)
        self.assertEqual(len(expr.pairs), 1)
        self.assertIsInstance(expr.pairs[0][0], Ps1SubExpression)

    def test_hash_literal_double_semicolon_separator(self):
        expr = self._parse_expr('@{ a = 1;; b = 2 }')
        self.assertIsInstance(expr, Ps1HashLiteral)
        self.assertEqual(len(expr.pairs), 2)

    def test_hash_literal_mixed_separators(self):
        expr = self._parse_expr("@{ a = 1;\n; b = 2;\n\n c = 3 }")
        self.assertIsInstance(expr, Ps1HashLiteral)
        self.assertEqual(len(expr.pairs), 3)

    def test_script_block(self):
        expr = self._parse_expr('{ $x + 1 }')
        self.assertIsInstance(expr, Ps1ScriptBlock)

    def test_assignment(self):
        expr = self._parse_expr('$x = 42')
        self.assertIsInstance(expr, Ps1AssignmentExpression)
        self.assertEqual(expr.operator, '=')

    def test_compound_assignment(self):
        expr = self._parse_expr('$x += 1')
        self.assertIsInstance(expr, Ps1AssignmentExpression)
        self.assertEqual(expr.operator, '+=')

    def test_format_operator(self):
        expr = self._parse_expr('"hello {0}" -f "world"')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '-f')

    def test_chained_member_access(self):
        expr = self._parse_expr('$s.Trim().ToLower()')
        self.assertIsInstance(expr, Ps1InvokeMember)
        self.assertEqual(expr.member, 'ToLower')
        self.assertIsInstance(expr.object, Ps1InvokeMember)

    def test_nested_index(self):
        expr = self._parse_expr('$a[0][1]')
        self.assertIsInstance(expr, Ps1IndexExpression)
        self.assertIsInstance(expr.object, Ps1IndexExpression)

    def test_unary_prefix_operators_allow_newline_before_operand(self):
        cases = {
            '-' : (' -\n    $y', '-'),
            '+' : (' +\n    $y', '+'),
            '!' : (' !\n    $y', '!'),
            '++': ('++\n    $y', '++'),
            '--': ('--\n    $y', '--'),
        }
        for label, (src, op) in cases.items():
            with self.subTest(operator=label):
                expr = self._parse_expr(src)
                self.assertIsInstance(expr, Ps1UnaryExpression)
                self.assertEqual(expr.operator, op)
                self.assertTrue(expr.prefix)
                self.assertIsInstance(expr.operand, Ps1Variable)
                self.assertEqual(expr.operand.name, 'y')

    def test_unary_comma_disabled_in_method_args(self):
        expr = self._parse_expr('$obj.Method($a, $b)')
        self.assertIsInstance(expr, Ps1InvokeMember)
        self.assertEqual(len(expr.arguments), 2)
        self.assertIsInstance(expr.arguments[0], Ps1Variable)
        self.assertIsInstance(expr.arguments[1], Ps1Variable)

    def test_unary_comma_not_parsed_in_method_args(self):
        # In a method argument context, a leading comma must not be parsed as unary array-wrap.
        p = Ps1Parser('$obj.Method(,$a)')
        script = p.parse()
        self.assertIsInstance(script, Ps1Script)
        stmt = script.body[0]
        self.assertIsInstance(stmt, Ps1ExpressionStatement)
        expr = stmt.expression
        self.assertIsInstance(expr, Ps1InvokeMember)
        for arg in expr.arguments:
            self.assertNotIsInstance(arg, Ps1ArrayLiteral)

    def test_hex_literal_with_multiplier_suffix(self):
        expr = self._parse_expr('0x10kb')
        self.assertIsInstance(expr, Ps1RealLiteral)
        self.assertAlmostEqual(expr.value, 16 * 1024)

    def test_binary_literal_with_multiplier_suffix(self):
        expr = self._parse_expr('0b1010mb')
        self.assertIsInstance(expr, Ps1RealLiteral)
        self.assertAlmostEqual(expr.value, 10 * 1024 ** 2)

    def test_hex_literal_with_gb_suffix(self):
        expr = self._parse_expr('0xFFgb')
        self.assertIsInstance(expr, Ps1RealLiteral)
        self.assertAlmostEqual(expr.value, 255 * 1024 ** 3)

    def test_type_literal_followed_by_comma_is_not_cast(self):
        # [int],1 should be an array of a type expression and an integer, not a cast.
        expr = self._parse_expr('[int], 1')
        self.assertIsInstance(expr, Ps1ArrayLiteral)
        self.assertEqual(len(expr.elements), 2)
        self.assertIsInstance(expr.elements[0], Ps1TypeExpression)
        self.assertEqual(expr.elements[0].name, 'int')
        self.assertIsInstance(expr.elements[1], Ps1IntegerLiteral)
        self.assertEqual(expr.elements[1].value, 1)

    def test_integer_with_long_suffix_and_multiplier(self):
        expr = self._parse_expr('10lkb')
        self.assertIsInstance(expr, Ps1RealLiteral)
        self.assertAlmostEqual(expr.value, 10 * 1024)

    def test_hex_integer_with_long_suffix_and_multiplier(self):
        expr = self._parse_expr('0xAlgb')
        self.assertIsInstance(expr, Ps1RealLiteral)
        self.assertAlmostEqual(expr.value, 10 * 1024 ** 3)

    def test_expandable_string_nested_dq_in_subexpr(self):
        expr = self._parse_expr('"value: $($x.ToString("N2"))"')
        self.assertIsInstance(expr, Ps1ExpandableString)
        has_subexpr = any(isinstance(p, Ps1SubExpression) for p in expr.parts)
        self.assertTrue(has_subexpr)

    def test_expandable_string_special_var_dollar_stops(self):
        expr = self._parse_expr('"$$x"')
        self.assertIsInstance(expr, Ps1ExpandableString)
        has_var = any(isinstance(p, Ps1Variable) and p.name == '$' for p in expr.parts)
        self.assertTrue(has_var, 'should contain variable $$ (name=$)')
        has_literal_x = any(
            isinstance(p, Ps1StringLiteral) and 'x' in p.value for p in expr.parts)
        self.assertTrue(has_literal_x, 'should contain literal x after $$')

    def test_digit_variable_standalone(self):
        for src, expected_name in [('$0', '0'), ('$1', '1'), ('$1foo', '1foo')]:
            with self.subTest(src=src):
                expr = self._parse_expr(src)
                self.assertIsInstance(expr, Ps1Variable)
                self.assertEqual(expr.name, expected_name)

    def test_digit_variable_in_expandable_string(self):
        expr = self._parse_expr('"text $1 end"')
        self.assertIsInstance(expr, Ps1ExpandableString)
        has_var = any(isinstance(p, Ps1Variable) and p.name == '1' for p in expr.parts)
        self.assertTrue(has_var, 'should contain variable $1 (name=1)')

    def test_expandable_string_special_var_question_stops(self):
        expr = self._parse_expr('"$?x"')
        self.assertIsInstance(expr, Ps1ExpandableString)
        has_var = any(isinstance(p, Ps1Variable) and p.name == '?' for p in expr.parts)
        self.assertTrue(has_var, 'should contain variable $? (name=?)')
        has_literal_x = any(
            isinstance(p, Ps1StringLiteral) and 'x' in p.value for p in expr.parts)
        self.assertTrue(has_literal_x, 'should contain literal x after $?')

    def test_here_string_verbatim_whitespace_after_header(self):
        # PowerShell allows whitespace (spaces/tabs) between the @' header and the newline.
        # The whitespace must not become part of the string content.
        expr = self._parse_expr("@'   \nline one\nline two\n'@")
        self.assertIsInstance(expr, Ps1HereString)
        self.assertEqual(expr.value, 'line one\nline two')

    def test_here_string_expandable_whitespace_after_header(self):
        expr = self._parse_expr('@"   \nline one\nline two\n"@')
        self.assertIsInstance(expr, Ps1HereString)
        self.assertEqual(expr.value, 'line one\nline two')

    def test_here_string_verbatim_tab_after_header(self):
        expr = self._parse_expr("@'\t\ntext\n'@")
        self.assertIsInstance(expr, Ps1HereString)
        self.assertEqual(expr.value, 'text')

    def test_here_string_verbatim_bare_cr_line_endings(self):
        expr = self._parse_expr("@'\rline one\rline two\r'@")
        self.assertIsInstance(expr, Ps1HereString)
        self.assertEqual(expr.value, 'line one\rline two')

    def test_here_string_expandable_bare_cr_line_endings(self):
        expr = self._parse_expr('@"\rline one\rline two\r"@')
        self.assertIsInstance(expr, Ps1HereString)
        self.assertEqual(expr.value, 'line one\rline two')

    def test_member_access_newline_after_dot(self):
        expr = self._parse_expr('$obj.\n    Length')
        self.assertIsInstance(expr, Ps1MemberAccess)
        self.assertEqual(expr.member, 'Length')

    def test_method_call_newline_after_dot(self):
        expr = self._parse_expr('$obj.\n    Method()')
        self.assertIsInstance(expr, Ps1InvokeMember)
        self.assertEqual(expr.member, 'Method')

    def test_static_access_newline_after_double_colon(self):
        expr = self._parse_expr('[int]::\n    MaxValue')
        self.assertIsInstance(expr, Ps1MemberAccess)
        self.assertEqual(expr.member, 'MaxValue')
        self.assertEqual(expr.access, Ps1AccessKind.STATIC)

    def test_cast_with_unary_not_operator(self):
        expr = self._parse_expr('[int]-not $false')
        self.assertIsInstance(expr, Ps1CastExpression)
        self.assertEqual(expr.type_name, 'int')
        self.assertIsInstance(expr.operand, Ps1UnaryExpression)
        self.assertEqual(expr.operand.operator, '-not')
        self.assertTrue(expr.operand.prefix)
        self.assertIsInstance(expr.operand.operand, Ps1Variable)
        self.assertEqual(expr.operand.operand.name, 'false')

    def test_chained_fluent_member_access(self):
        expr = self._parse_expr('$s.\n    Trim().\n    ToLower()')
        self.assertIsInstance(expr, Ps1InvokeMember)
        self.assertEqual(expr.member, 'ToLower')
        self.assertIsInstance(expr.object, Ps1InvokeMember)
        self.assertEqual(expr.object.member, 'Trim')

    def test_unicode_dash_operators(self):
        for dash in '\u2013\u2014\u2015':
            with self.subTest(dash=F'U+{ord(dash):04X}'):
                expr = self._parse_expr(F'$x {dash}eq 1')
                self.assertIsInstance(expr, Ps1BinaryExpression)
                self.assertEqual(expr.operator, '-eq')

    def test_unicode_double_quote_expandable_with_variable(self):
        for oq, cq in itertools.product('\u201C\u201D\u201E', repeat=2):
            with self.subTest(oq=F'U+{ord(oq):04X}', cq=F'U+{ord(cq):04X}'):
                expr = self._parse_expr(F'{oq}hello $name{cq}')
                self.assertIsInstance(expr, Ps1ExpandableString)
                self.assertTrue(len(expr.parts) >= 2)

    def test_unicode_single_quote_escape(self):
        for q in '\u2018\u2019\u201A\u201B':
            with self.subTest(q=F'U+{ord(q):04X}'):
                expr = self._parse_expr(F"{q}it{q}{q}s here{q}")
                self.assertIsInstance(expr, Ps1StringLiteral)
                self.assertEqual(expr.value, "it's here")

    def test_unicode_double_quote_escape(self):
        for q in '\u201C\u201D\u201E':
            with self.subTest(q=F'U+{ord(q):04X}'):
                expr = self._parse_expr(F'{q}say {q}{q}hi{q}{q}{q}')
                self.assertIsInstance(expr, Ps1StringLiteral)
                self.assertEqual(expr.value, 'say "hi"')

    def test_unicode_whitespace(self):
        # no-break space (U+00A0) should act as whitespace between tokens
        expr = self._parse_expr('1\u00A0+\u00A02')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '+')

    def test_unicode_decrement_operator(self):
        for d1, d2 in itertools.product('\u2013\u2014\u2015', repeat=2):
            with self.subTest(d1=F'U+{ord(d1):04X}', d2=F'U+{ord(d2):04X}'):
                expr = self._parse_expr(F'{d1}{d2}$x')
                self.assertIsInstance(expr, Ps1UnaryExpression)
                self.assertEqual(expr.operator, '--')

    def test_inline_block_comment_before_operator(self):
        expr = self._parse_expr('1 <# comment #> + 2')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '+')

    def test_inline_block_comment_after_operator(self):
        expr = self._parse_expr('1 + <# comment #> 2')
        self.assertIsInstance(expr, Ps1BinaryExpression)
        self.assertEqual(expr.operator, '+')

    def test_paren_command_static_member_access(self):
        expr = self._parse_expr('(Get-Variable Y -ValueOnly)::Tls')
        self.assertIsInstance(expr, Ps1MemberAccess)
        self.assertEqual(expr.access, Ps1AccessKind.STATIC)
        self.assertIsInstance(expr.object, Ps1ParenExpression)
        self.assertEqual(expr.member, 'Tls')

    def test_paren_command_instance_member_access(self):
        expr = self._parse_expr('(Get-Item Variable:X).Value')
        self.assertIsInstance(expr, Ps1MemberAccess)
        self.assertEqual(expr.access, Ps1AccessKind.INSTANCE)
        self.assertIsInstance(expr.object, Ps1ParenExpression)
        self.assertEqual(expr.member, 'Value')

    def test_paren_command_static_string_member(self):
        expr = self._parse_expr("(Get-Variable Y -ValueOnly)::\"T`Ls\"")
        self.assertIsInstance(expr, Ps1MemberAccess)
        self.assertEqual(expr.access, Ps1AccessKind.STATIC)
        self.assertIsInstance(expr.member, Ps1StringLiteral)

    def test_leading_zero_integer_is_decimal(self):
        expr = self._parse_expr('007')
        self.assertIsInstance(expr, Ps1IntegerLiteral)
        self.assertEqual(expr.value, 7)

    def test_backtick_unicode_escape(self):
        expr = self._parse_expr('"`u{48}`u{69}"')
        self.assertIsInstance(expr, Ps1StringLiteral)
        self.assertEqual(expr.value, 'Hi')

    def test_verbatim_here_string_keeps_smart_quotes(self):
        content = 'say “hi” don’t'
        expr = self._parse_expr(F"@'\n{content}\n'@")
        self.assertIsInstance(expr, Ps1HereString)
        self.assertEqual(expr.value, content)

    def test_generic_argument_token_is_decoded(self):
        # The generic argument token `a'b c'd` decodes its embedded quotes to the value `ab cd`.
        script = Ps1Parser("echo a'b c'd").parse()
        values = [n.value for n in script.walk() if isinstance(n, Ps1StringLiteral)]
        self.assertEqual(values, ['ab cd', 'echo'])


class TestPs1MemberAccessBindsOnlyToWhatItTouches(TestBase):
    """
    A member access is written with the dot against its object: 5.1 reads `$a.Length` as a property
    and `$a . Length` as three elements of a command, and it refuses `('a')    .('Length')` outright
    with `Unexpected token '.'`. A block comment is not whitespace and separates nothing.

    A dot that binds across whitespace turns a script 5.1 refuses to run, or runs as a command, into
    a property read that looks deliberate, and nothing in the output marks the difference.
    """

    @staticmethod
    def _parse(source: str) -> Ps1Script:
        return Ps1Parser(source).parse()

    @staticmethod
    def _member_accesses(script: Ps1Script) -> list[Ps1MemberAccess | Ps1InvokeMember]:
        return [
            node for node in script.walk()
            if isinstance(node, (Ps1MemberAccess, Ps1InvokeMember))
        ]

    def _assertLengthIsAPropertyOfTheVariable(self, source: str):
        accesses = self._member_accesses(self._parse(source))
        self.assertEqual(len(accesses), 1, source)
        access = accesses[0]
        self.assertIsInstance(access, Ps1MemberAccess)
        self.assertIsInstance(access.object, Ps1Variable)
        self.assertEqual(access.object.name, 'a')
        self.assertEqual(access.member, 'Length')

    def test_a_dot_touching_the_object_and_the_member_is_a_member_access(self):
        self._assertLengthIsAPropertyOfTheVariable('$a.Length')

    def test_a_block_comment_does_not_separate_the_member_from_its_object(self):
        self._assertLengthIsAPropertyOfTheVariable('$a<# does this separate? #>.Length')

    def test_a_dot_touching_the_object_and_the_method_is_a_method_call(self):
        calls = self._member_accesses(self._parse('$a.Substring(1)'))
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], Ps1InvokeMember)
        self.assertEqual(calls[0].member, 'Substring')

    def test_whitespace_before_the_dot_leaves_the_member_a_word_of_its_own(self):
        """
        `$a . Length` reads no property: `$a` keeps the value it had and `Length` stands on its own.
        The three elements are asserted rather than the command 5.1 makes of them, because the
        parser groups them differently, and how a lone dot is grouped is a defect of its own.
        """
        script = self._parse('$a . Length')
        variables = [node.name for node in script.walk() if isinstance(node, Ps1Variable)]
        words = [node.value for node in script.walk() if isinstance(node, Ps1StringLiteral)]
        self.assertEqual(
            self._member_accesses(script), [], 'the spaced dot bound the word to the variable')
        self.assertEqual(variables, ['a'])
        self.assertEqual(words, ['Length'])

    def test_whitespace_before_the_dot_of_a_method_call_reads_no_member(self):
        self.assertEqual(self._member_accesses(self._parse('$a .Substring(1)')), [])

    def test_a_block_comment_does_not_separate_a_computed_member_from_its_object(self):
        accesses = self._member_accesses(self._parse("('a')<# does this separate? #>.('Length')"))
        self.assertEqual(len(accesses), 1)
        access = accesses[0]
        self.assertIsInstance(access, Ps1MemberAccess)
        self.assertIsInstance(access.object, Ps1ParenExpression)
        self.assertIsInstance(access.member, Ps1ParenExpression)

    def test_whitespace_before_a_computed_member_reads_no_member(self):
        """
        5.1 refuses `('a')    .('Length')` with `Unexpected token '.'`, so nothing in that script
        runs at all; a member read there is one no execution of it ever performs.
        """
        self.assertEqual(self._member_accesses(self._parse("('a')    .('Length')")), [])

    def test_whitespace_behind_the_dot_does_not_separate_the_member_from_its_object(self):
        """
        The rule is about what stands before an operator: 5.1 reads `$a.  Length` as the property
        that `$a.Length` reads, so the gap behind the dot separates nothing.
        """
        self._assertLengthIsAPropertyOfTheVariable('$a.  Length')

    def test_whitespace_behind_the_double_colon_does_not_separate_the_member_from_its_type(self):
        accesses = self._member_accesses(self._parse('[int]::  MaxValue'))
        self.assertEqual(len(accesses), 1)
        access = accesses[0]
        self.assertIsInstance(access, Ps1MemberAccess)
        self.assertIsInstance(access.object, Ps1TypeExpression)
        self.assertEqual(access.object.name, 'int')
        self.assertEqual(access.member, 'MaxValue')
        self.assertEqual(access.access, Ps1AccessKind.STATIC)

    def test_a_parenthesis_touching_the_member_calls_it(self):
        script = self._parse('$a.Length()')
        self.assertEqual(len(script.body), 1)
        calls = self._member_accesses(script)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIsInstance(call, Ps1InvokeMember)
        self.assertIsInstance(call.object, Ps1Variable)
        self.assertEqual(call.object.name, 'a')
        self.assertEqual(call.member, 'Length')
        self.assertEqual(call.arguments, [])

    def test_whitespace_before_the_parenthesis_leaves_the_member_a_property_read(self):
        """
        5.1 reads `$a.Length ()` as the property `$a.Length` and a parenthesis of its own, so a
        method call read there invents a call that no execution of the script performs.
        """
        self._assertLengthIsAPropertyOfTheVariable('$a.Length ()')
        script = self._parse('$a.Length ()')
        self.assertEqual(len(script.body), 2)
        second = script.body[1]
        self.assertIsInstance(second, Ps1ExpressionStatement)
        self.assertIsInstance(second.expression, Ps1ParenExpression)

    def test_whitespace_before_the_parenthesis_of_a_static_call_reads_no_call(self):
        accesses = self._member_accesses(self._parse("[Convert]::FromBase64String ('AA==')"))
        self.assertEqual(len(accesses), 1)
        self.assertIsInstance(accesses[0], Ps1MemberAccess)
        self.assertEqual(accesses[0].member, 'FromBase64String')


class TestPs1ALineContinuationSeparatesTheOperatorFromWhatPrecedesIt(TestBase):
    """
    A backtick before a newline is a line continuation and a member operator binds only to what it
    touches, so a continuation between the two separates them. 5.1 refuses the type expression
    `[Convert]`, a continuation and `::FromBase64String('AA==')` with a parse error and reads it as
    two pipelines, the type expression and then a command; `$a`, a continuation and `.Length` as
    well as `$a`, a continuation and `[0]` read the same way. Written adjacent, all three bind.

    A line continuation is a common obfuscation device, so a parser that folds one away reads a
    member of an object the script never reaches and resolves a call 5.1 never makes.
    """

    @staticmethod
    def _parse(source: str) -> Ps1Script:
        return Ps1Parser(source).parse()

    @staticmethod
    def _bindings(script: Ps1Script) -> list[Node]:
        return [
            node for node in script.walk()
            if isinstance(node, (Ps1MemberAccess, Ps1InvokeMember, Ps1IndexExpression))
        ]

    def _assertTheObjectIsAStatementOfItsOwn(self, script: Ps1Script, expected: type):
        self.assertGreater(len(script.body), 1)
        first = script.body[0]
        self.assertIsInstance(first, Ps1ExpressionStatement)
        self.assertIsInstance(first.expression, expected)
        return first.expression

    def test_a_continuation_before_a_static_member_leaves_the_type_expression_alone(self):
        script = self._parse("[Convert]`\n::FromBase64String('AA==')")
        self.assertEqual(self._bindings(script), [], 'the continuation bound the member to a type')
        expression = self._assertTheObjectIsAStatementOfItsOwn(script, Ps1TypeExpression)
        self.assertEqual(expression.name, 'Convert')
        words = [node.value for node in script.walk() if isinstance(node, Ps1StringLiteral)]
        self.assertIn('AA==', words, 'the text behind the continuation was lost')

    def test_a_continuation_before_a_property_leaves_the_variable_alone(self):
        script = self._parse('$a`\n.Length')
        self.assertEqual(self._bindings(script), [], 'the continuation bound the member to $a')
        expression = self._assertTheObjectIsAStatementOfItsOwn(script, Ps1Variable)
        self.assertEqual(expression.name, 'a')
        words = [node.value for node in script.walk() if isinstance(node, Ps1StringLiteral)]
        self.assertTrue(
            any('Length' in word for word in words), 'the text behind the continuation was lost')

    def test_a_continuation_before_an_index_leaves_the_variable_alone(self):
        script = self._parse('$a`\n[0]')
        self.assertEqual(self._bindings(script), [], 'the continuation indexed $a')
        expression = self._assertTheObjectIsAStatementOfItsOwn(script, Ps1Variable)
        self.assertEqual(expression.name, 'a')

    def test_a_static_member_touching_its_type_is_a_call(self):
        script = self._parse("[Convert]::FromBase64String('AA==')")
        bindings = self._bindings(script)
        self.assertEqual(len(bindings), 1)
        call = bindings[0]
        self.assertIsInstance(call, Ps1InvokeMember)
        self.assertIsInstance(call.object, Ps1TypeExpression)
        self.assertEqual(call.object.name, 'Convert')
        self.assertEqual(call.member, 'FromBase64String')

    def test_a_property_touching_its_variable_is_a_member_access(self):
        script = self._parse('$a.Length')
        bindings = self._bindings(script)
        self.assertEqual(len(bindings), 1)
        access = bindings[0]
        self.assertIsInstance(access, Ps1MemberAccess)
        self.assertIsInstance(access.object, Ps1Variable)
        self.assertEqual(access.member, 'Length')

    def test_an_index_touching_its_variable_is_an_index_access(self):
        script = self._parse('$a[0]')
        bindings = self._bindings(script)
        self.assertEqual(len(bindings), 1)
        index = bindings[0]
        self.assertIsInstance(index, Ps1IndexExpression)
        self.assertIsInstance(index.object, Ps1Variable)
        self.assertIsInstance(index.index, Ps1IntegerLiteral)
        self.assertEqual(index.index.value, 0)

    def test_a_continuation_written_with_a_windows_line_ending_separates_as_well(self):
        """
        The scripts this device is found in carry CRLF line endings, so the continuation has to
        separate there exactly as it does with a bare newline.
        """
        for source in [
            "[Convert]`\r\n::FromBase64String('AA==')",
            '$a`\r\n.Length',
            '$a`\r\n[0]',
        ]:
            with self.subTest(source=source):
                script = self._parse(source)
                self.assertEqual(self._bindings(script), [])
                self.assertGreater(len(script.body), 1)

    def test_a_continuation_leaves_the_static_call_one_pipeline_of_its_own(self):
        """
        The tree 5.1 recovers from the parse error holds two pipelines, the type expression and one
        command that keeps the parenthesis behind it.
        """
        script = self._parse("[Convert]`\n::FromBase64String('AA==')")
        self.assertEqual(len(script.body), 2)
        second = script.body[1]
        self.assertIsInstance(second, Ps1ExpressionStatement)
        self.assertIsInstance(second.expression, Ps1CommandInvocation)
