from __future__ import annotations

import unittest
import unittest.mock

from refinery.lib.scripts.ps1.deobfuscation import _folds, deobfuscate
from refinery.lib.scripts.ps1.parser import Ps1Parser
from refinery.lib.scripts.ps1.synth import Ps1Synthesizer

from test.lib.scripts.ps1.corpus import oracle_corpus
from test.lib.scripts.ps1.deobfuscation.fold_census import FOLDS


def census() -> dict[str, str]:
    """
    Every corpus row the deobfuscator rewrites, with what it rewrites it to. A row whose output
    equals its canonical input is absent rather than mapped to itself, so the population of this
    dictionary is the set of folds the tool takes.
    """
    folded = {}
    for row in oracle_corpus():
        ast = Ps1Parser(row).parse()
        before = Ps1Synthesizer().convert(ast)
        deobfuscate(ast)
        after = Ps1Synthesizer().convert(ast)
        if after != before:
            folded[row] = after
    return folded


#: What each folding pass contributes to the census: the rows that stop being folded at all when it
#: is neutered, and the rows that still fold but to something else. Two passes contribute nothing,
#: which is a fact about the corpus rather than about them — it holds no flattened control flow and
#: no expandable string that survives hoisting into the output — and they are pinned at zero so that
#: a corpus row which starts exercising one is noticed.
_CONTRIBUTION: dict[str, tuple[int, int]] = {
    'Ps1ConstantFolding': (75, 60),
    'Ps1DeadCodeElimination': (1, 12),
    'Ps1ControlFlowDeflattening': (0, 0),
    'Ps1ConstantInlining': (67, 153),
    'Ps1ExpandableStringHoist': (0, 0),
    'Ps1TypeCasts': (39, 12),
}


class TestPs1FoldCensus(unittest.TestCase):
    """
    Which folds the deobfuscator takes over the whole corpus, held against a recorded baseline.

    This is the instrument that makes *losing* a fold visible. A change that refuses more than it
    used to still passes every test that asserts an answer is correct, because a refusal is not a
    wrong answer — it only stops being an answer. Here it is a key that went missing.
    """

    def test_every_fold_the_corpus_takes_is_the_one_recorded(self):
        self.assertEqual(census(), FOLDS)

    def test_a_neutered_pass_is_seen_by_the_census(self):
        """
        The census can only be trusted to catch a lost fold if a lost fold moves it. Each folding
        pass is disabled in turn and the census re-taken; what the pass was contributing is the
        difference, and it is pinned so that a pass which quietly stops contributing is a failure
        rather than a smaller number nobody reads.
        """
        baseline = census()
        for transformer in _folds:
            with self.subTest(transformer.__name__):
                with unittest.mock.patch.object(transformer, 'visit', lambda self, node: None):
                    without = census()
                lost = set(baseline) - set(without)
                differing = {
                    row for row in set(baseline) & set(without)
                    if baseline[row] != without[row]
                }
                self.assertEqual(
                    (len(lost), len(differing)), _CONTRIBUTION[transformer.__name__])

    def test_no_pass_folds_a_row_the_census_does_not_hold(self):
        """
        Neutering a pass may only take folds away. One that *adds* a fold when it is disabled means
        the passes are fighting, and the census would record whichever won the race.
        """
        baseline = census()
        for transformer in _folds:
            with self.subTest(transformer.__name__):
                with unittest.mock.patch.object(transformer, 'visit', lambda self, node: None):
                    without = census()
                self.assertEqual(sorted(set(without) - set(baseline)), [])
