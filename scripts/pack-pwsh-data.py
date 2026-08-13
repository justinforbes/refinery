#!/usr/bin/env python3
"""
Pack captured PowerShell data into the compressed resources that ship inside the package.

`run-pwsh.ps1` and `run-pwsh-operators.ps1` each write plain `pwsh-*.json` into an output directory.
Turning those into the `pwsh-*.json.xz` that `refinery.lib.scripts.ps1.data` loads had been a step
nobody had written down, and it had already been performed two different ways: the five host tables
were compacted on the way in and the operator grid was not, which is why one shipped resource was
86% indentation and six times the size of the same data.

So the step is this script. It re-serializes rather than trusting what it is given, which makes the
size a property of the packing and not of whether a capture script remembered a flag, and the
container it writes carries no timestamp, so packing the same capture twice produces the same bytes.
"""
from __future__ import annotations

import argparse
import json
import lzma
import pathlib
import sys


#: What the resources are compressed with. LZMA at its strongest preset, because these tables are
#: long stretches of repeated key names and type names and a large window is what pays off on them:
#: it halves what gzip achieves, and the extra time it costs to decompress is a fraction of what
#: parsing the result costs either way.
_FILTERS = [{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]


def pack(source: pathlib.Path) -> bytes:
    """
    The bytes a captured file ships as: its own content, compactly re-serialized and compressed.

    The document is parsed before it is written, because a resource that does not parse is one the
    package cannot load at all, and a packing step is the last place that can still be noticed. It
    is not checked for a schema version: only `pwsh-meta` and `pwsh-operators` carry one, and the
    four tables beside them are versioned by the `pwsh-meta` written in the same capture.
    """
    document = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise ValueError(F'{source.name} is a {type(document).__name__}, not a captured table.')
    text = json.dumps(document, separators=(',', ':'), ensure_ascii=False)
    return lzma.compress(text.encode('utf-8'), format=lzma.FORMAT_XZ, filters=_FILTERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'source',
        type=pathlib.Path,
        help='directory holding the captured pwsh-*.json files',
    )
    parser.add_argument(
        '-o', '--output',
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / 'refinery' / 'data',
        help='directory the compressed resources are written to',
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='report what each file would pack to without writing anything',
    )
    args = parser.parse_args(argv)

    captured = sorted(args.source.glob('pwsh-*.json'))
    if not captured:
        parser.error(F'no pwsh-*.json files in {args.source}')

    for source in captured:
        if source.name.endswith('.failed.json'):
            continue
        blob = pack(source)
        target = args.output / F'{source.name}.xz'
        before = target.stat().st_size if target.exists() else 0
        if not args.dry_run:
            target.write_bytes(blob)
        print(F'{target.name:26} {before:>9,} -> {len(blob):>9,}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
