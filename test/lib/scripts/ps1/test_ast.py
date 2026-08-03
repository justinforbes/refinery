from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.ps1.ast import in_evaluation_order
from refinery.lib.scripts.ps1.model import Ps1StringLiteral, Ps1Variable
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
