"""
The `--no-pin` differential: run any test selection with every model-cache pin neutralized, so each
model read observes the current tree. A pin is a cost device, never a semantic one — the obligation
`refinery.lib.scripts.modelcache.ModelCacheBase.pinned` states — so any test that fails only under
this option has found a pass acting on a stale answer. Build-count pins measure the cost the pin
itself buys and skip themselves through the `REFINERY_TEST_NO_PIN` environment variable this option
sets before collection.
"""
from __future__ import annotations

import os

from contextlib import contextmanager


def pytest_addoption(parser):
    parser.addoption(
        '--no-pin',
        action='store_true',
        default=False,
        help='neutralize every model-cache pin and run the selection as a differential',
    )


def pytest_configure(config):
    if not config.getoption('--no-pin'):
        return
    os.environ['REFINERY_TEST_NO_PIN'] = '1'
    from refinery.lib.scripts.modelcache import ModelCacheBase

    @contextmanager
    def unpinned(self):
        yield self

    ModelCacheBase.pinned = unpinned
