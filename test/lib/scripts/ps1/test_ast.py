from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import Node
from refinery.lib.scripts.ps1.ast import (
    bound_argument_value,
    in_evaluation_order,
    is_reference_cast,
)
from refinery.lib.scripts.ps1.model import (
    Ps1CastExpression,
    Ps1CommandInvocation,
    Ps1StringLiteral,
    Ps1Variable,
)
from refinery.lib.scripts.ps1.parser import Ps1Parser


class TestPs1EvaluationOrder(TestBase):
    """
    The order PowerShell evaluates the parts of one statement in, which is where a read and a write
    of the same name sitting at one control-flow node are told apart.
    """

    @staticmethod
    def _order(source: str) -> list[str]:
        """
        The variables and string literals of *source*, in evaluation order, a variable rendered with
        its `$`. Only these two forms carry a name, and every case below turns on their order.
        """
        rendered: list[str] = []
        for node in in_evaluation_order(Ps1Parser(source).parse()):
            if isinstance(node, Ps1Variable):
                rendered.append(F'${node.name}')
            elif isinstance(node, Ps1StringLiteral):
                rendered.append(node.value)
        return rendered

    def test_an_assignment_produces_its_value_before_it_stores_it(self):
        """
        Why `$x = [char]($x)` reads the previous `$x`: the target is written first and stored last.
        """
        self.assertEqual(self._order("$x = $y"), ['$y', '$x'])

    def test_arguments_are_evaluated_left_to_right(self):
        self.assertEqual(self._order("f $a 'lit' $b"), ['f', '$a', 'lit', '$b'])

    def test_a_nested_assignment_stores_before_the_one_holding_it(self):
        self.assertEqual(self._order("$x = ($y = $z)"), ['$z', '$y', '$x'])

    def test_the_operands_of_a_binary_expression_keep_their_order(self):
        self.assertEqual(self._order("$y = $a + $b"), ['$a', '$b', '$y'])

    def test_a_multi_assignment_stores_every_target_after_the_whole_value(self):
        """
        `$x, $y = $y, $x` swaps, so neither target may be ordered before either source.
        """
        self.assertEqual(self._order("$x, $y = $y, $x"), ['$y', '$x', '$x', '$y'])


class TestPs1ReferenceCast(TestBase):
    """
    The `[ref]` recognizer, which decides whether a callee is handed storage it can write back
    through rather than a value. Its answer is what makes the difference between a name a store may
    be folded across and one it may not.
    """

    @staticmethod
    def _cast(source: str) -> Node:
        for node in Ps1Parser(source).parse().walk():
            if isinstance(node, Ps1CastExpression):
                return node
        raise AssertionError(F'no cast in {source!r}')

    def test_both_spellings_of_the_wrapper_type_are_recognized(self):
        for source in (
            '[ref]$n',
            '[Ref]$n',
            '[REF]$n',
            '[management.automation.psreference]$n',
            '[System.Management.Automation.PSReference]$n',
        ):
            with self.subTest(source):
                self.assertTrue(is_reference_cast(self._cast(source)))

    def test_an_unrelated_cast_is_not_a_reference(self):
        for source in ('[int]$n', '[string]$n', '[scriptblock]$n', '[refx]$n'):
            with self.subTest(source):
                self.assertFalse(is_reference_cast(self._cast(source)))

    def test_something_that_is_not_a_cast_is_not_a_reference(self):
        node = Ps1Parser('$n').parse()
        self.assertFalse(is_reference_cast(node))
        self.assertFalse(is_reference_cast(None))


class TestPs1BoundArgumentValue(TestBase):
    """
    Which value a command binds to a named parameter. PowerShell decides this from the command's own
    parameter metadata, which the parser does not have — it leaves `-Name x` as a switch followed by
    a positional, exactly as it leaves `-Recurse C:\\` — so the accessor has to reconstruct it and
    the caller has to know the parameter takes a value.
    """

    @staticmethod
    def _command(source: str) -> Ps1CommandInvocation:
        for node in Ps1Parser(source).parse().walk():
            if isinstance(node, Ps1CommandInvocation):
                return node
        raise AssertionError(F'no command in {source!r}')

    def _value(self, source: str, parameter: str) -> str | None:
        found = bound_argument_value(self._command(source), parameter)
        return None if found is None else found.value

    def test_both_spellings_of_a_binding_are_found(self):
        for source in (
            'Set-Variable -Name x -Value 5',
            'Set-Variable -Name:x -Value:5',
        ):
            with self.subTest(source):
                self.assertEqual(self._value(source, 'name'), 'x')

    def test_an_abbreviation_binds_the_parameter_it_abbreviates(self):
        for source in ('Set-Variable -Na x', 'Set-Variable -N x', 'Set-Variable -Nam:x'):
            with self.subTest(source):
                self.assertEqual(self._value(source, 'name'), 'x')

    def test_a_longer_parameter_that_merely_starts_the_same_does_not_bind(self):
        """
        `-Namespace` is not an abbreviation of `-Name`; the abbreviation relation runs the other way,
        and testing it backwards binds every parameter whose name begins with this one's.
        """
        self.assertIsNone(self._value('Set-Variable -Namespace x', 'name'))

    def test_a_parameter_that_is_not_written_binds_nothing(self):
        self.assertIsNone(self._value('Set-Variable -Value 5', 'name'))

    def test_the_append_form_of_a_name_keeps_its_marker(self):
        """
        `-OutVariable +p` appends to `$p` and reads its previous value where `-OutVariable p`
        replaces it, so the `+` has to survive to the caller that tells the two apart.
        """
        self.assertEqual(self._value('Get-Process -OutVariable +p', 'outvariable'), '+p')
        self.assertEqual(self._value('Get-Process -OutVariable p', 'outvariable'), 'p')

    def test_an_alias_of_a_parameter_binds_it_only_as_its_own_name(self):
        self.assertEqual(self._value('Get-Process -ov p', 'ov'), 'p')
        self.assertIsNone(self._value('Get-Process -ov p', 'outvariable'))
