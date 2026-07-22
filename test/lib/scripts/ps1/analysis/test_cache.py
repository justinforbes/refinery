from __future__ import annotations

from test import TestBase

from refinery.lib.scripts import Transformer, _remove_from_parent, set_body
from refinery.lib.scripts.ps1.analysis.cache import Ps1ModelCache, model_cache
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

    def test_mutating_an_unrelated_tree_keeps_the_cached_model(self):
        cache = Ps1ModelCache(self._script("$a = 1\n$b = 2"))
        first = cache.model
        unrelated = self._script("$c = 3\n$d = 4")
        _remove_from_parent(unrelated.body[0])
        self.assertIs(cache.model, first)

    def test_body_splice_advances_the_version_and_rebuilds(self):
        # A whole-body rewrite through the counter-bumping splice helper must invalidate the cache the
        # same way a per-node removal does.
        script = self._script("$a = 1\n$b = 2")
        cache = Ps1ModelCache(script)
        first = cache.model
        set_body(script, list(script.body[:1]))
        self.assertIsNot(cache.model, first)

    def test_model_cache_reuses_the_stash_by_root_identity(self):
        script = self._script("$a = 1")
        transformer = Transformer()
        first = model_cache(transformer, script)
        self.assertIs(model_cache(transformer, script), first)

    def test_model_cache_rebuilds_for_a_different_root(self):
        transformer = Transformer()
        first = model_cache(transformer, self._script("$a = 1"))
        self.assertIsNot(model_cache(transformer, self._script("$b = 2")), first)
