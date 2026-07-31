from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import Transformer, _remove_from_parent, set_body, set_child_list
from refinery.lib.scripts.ps1.analysis.cache import Ps1ModelCache, model_cache
from refinery.lib.scripts.ps1.model import Ps1IfStatement
from refinery.lib.scripts.ps1.parser import Ps1Parser


class TestPs1ModelCache(TestBase):

    @staticmethod
    def _script(source: str):
        return Ps1Parser(source).parse()

    def test_model_is_memoized_while_the_tree_is_unchanged(self):
        cache = Ps1ModelCache(self._script("$a = 1\n$b = 2"))
        first = cache.model
        self.assertIs(cache.model, first)

    def test_mutating_the_cached_tree_rebuilds_the_model(self):
        script = self._script("$a = 1\n$b = 2")
        cache = Ps1ModelCache(script)
        first = cache.model
        _remove_from_parent(script.body[0])
        self.assertIsNot(cache.model, first)

    def test_cycles_are_memoized_while_the_tree_is_unchanged(self):
        cache = Ps1ModelCache(self._script("while ($x) { $a = 1 }"))
        first = cache.cycles
        self.assertIs(cache.cycles, first)

    def test_mutating_the_cached_tree_rebuilds_the_cycles(self):
        """
        The cycle sets are read off the control-flow graphs, which are keyed to node identity, so a
        model kept across a mutation would answer about statements the tree no longer holds.
        """
        script = self._script("while ($x) { $a = 1 }\n$b = 2")
        cache = Ps1ModelCache(script)
        first = cache.cycles
        _remove_from_parent(script.body[1])
        self.assertIsNot(cache.cycles, first)

    def test_mutating_an_unrelated_tree_keeps_the_cached_model(self):
        cache = Ps1ModelCache(self._script("$a = 1\n$b = 2"))
        first = cache.model
        unrelated = self._script("$c = 3\n$d = 4")
        _remove_from_parent(unrelated.body[0])
        self.assertIs(cache.model, first)

    def test_body_splice_advances_the_version_and_rebuilds(self):
        # A whole-body rewrite through the counter-bumping splice helper must invalidate the
        # cache the same way a per-node removal does.
        script = self._script("$a = 1\n$b = 2")
        cache = Ps1ModelCache(script)
        first = cache.model
        set_body(script, list(script.body[:1]))
        self.assertIsNot(cache.model, first)

    def test_body_splice_keeps_the_list_object(self):
        # The splice is in place: a transform that kept a reference to the list it was handed
        # must go on observing the node's current children rather than a detached snapshot.
        script = self._script("$a = 1\n$b = 2")
        body = script.body
        set_body(script, list(script.body[:1]))
        self.assertIs(script.body, body)
        self.assertEqual(len(body), 1)

    def test_clause_splice_adopts_the_nodes_inside_tuples(self):
        # An if-statement clause is a (condition, block) tuple; both halves are children of the
        # statement and must be adopted by a splice of the clause list.
        script = self._script('if ($a) { $b = 1 } elseif ($c) { $d = 2 }')
        statement = script.body[0]
        assert isinstance(statement, Ps1IfStatement)
        condition, block = statement.clauses[1]
        set_child_list(statement, 'clauses', [(condition, block)])
        self.assertIs(condition.parent, statement)
        self.assertIs(block.parent, statement)

    def test_model_cache_reuses_the_stash_by_root_identity(self):
        script = self._script("$a = 1")
        transformer = Transformer()
        first = model_cache(transformer, script)
        self.assertIs(model_cache(transformer, script), first)

    def test_model_cache_rebuilds_for_a_different_root(self):
        transformer = Transformer()
        first = model_cache(transformer, self._script("$a = 1"))
        self.assertIsNot(model_cache(transformer, self._script("$b = 2")), first)
