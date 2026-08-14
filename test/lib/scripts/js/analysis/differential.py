"""
Differential testing support for JavaScript deobfuscation: run a snippet and its deobfuscated form in
a real Node.js engine and compare observable behavior. This is the strongest available oracle for the
invariant that deobfuscation preserves semantics — the engine, not our own interpreter, decides.

Two execution models are available, because they disagree about what a top-level declaration means.
`behavior` runs the snippet as `node <file>`, a CommonJS module, where a top-level `var`/`function` is
scoped to the module. `host_behavior` runs it as a classic global script and can additionally call a
name through `globalThis` afterwards, which is the only way to observe a declaration that reaches the
global object — and therefore the only way to observe whether an entrypoint a host would call survived.

SECURITY: this executes JavaScript in Node.js. It must only ever be given benign, hand-authored
snippets. Never point it at the repository's malware test corpus or any untrusted sample — executing
those is forbidden.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path
from typing import Sequence

from refinery.lib.scripts.js.deobfuscation import deobfuscate
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.synth import JsSynthesizer

_ERROR_RE = re.compile(r'^([A-Za-z]+Error): .*$', re.MULTILINE)


def node_executable() -> str | None:
    """
    The path to the Node.js executable, or `None` when it is not installed.
    """
    return shutil.which('node')


def deobfuscate_source(
    source: str,
    *,
    module: bool = False,
    entrypoints: tuple[str, ...] = (),
) -> str:
    """
    Parse, deobfuscate, and re-synthesize a snippet, returning the deobfuscated source. *module*
    selects the module execution model (the oracle runs each snippet as a CommonJS module, so a
    scope-sensitive snippet must be deobfuscated under the same model to be comparable).
    *entrypoints* names top-level functions a host calls by name, which intra-script reachability
    cannot see.
    """
    ast = JsParser(source).parse()
    deobfuscate(ast, module=module, entrypoints=entrypoints)
    return JsSynthesizer().convert(ast)


_DEOBFUSCATE_IN_CHILD = R'''
import sys
sys.path.insert(0, sys.argv[1])
from test.lib.scripts.js.analysis.differential import deobfuscate_source
source = sys.stdin.buffer.read().decode('utf-8')
sys.stdout.buffer.write(deobfuscate_source(source).encode('utf-8'))
'''


class DeobfuscationFailed(Exception):
    """
    The child process running a deobfuscation exited without producing one. It carries the child's
    standard error, which is where the traceback is.
    """


def deobfuscate_within(source: str, seconds: float) -> str | None:
    """
    The deobfuscation of *source*, or `None` when it did not finish within *seconds*.

    Termination is a property worth asserting on its own: a fold that computes in unbounded integer
    arithmetic can run for hours on an expression a double answers in one operation, and that is a
    defect no comparison of results can express. It runs in a child process because such a computation
    happens inside a single interpreter opcode, where no timer, signal, or thread can interrupt it —
    only killing the process can.

    Both the source and the result cross the process boundary as UTF-8 bytes over a pipe rather than
    as text through the platform's codec, so that a program the console encoding cannot spell is
    reported as what it is. The child is killed on every way out and not only on the timeout,
    because the one thing this helper must never do is leave behind the runaway process it exists to
    bound.
    """
    root = Path(__file__).resolve().parents[5]
    with subprocess.Popen(
        [sys.executable, '-c', _DEOBFUSCATE_IN_CHILD, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as child:
        try:
            out, err = child.communicate(source.encode('utf-8'), timeout=seconds)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            return None
        except BaseException:
            child.kill()
            raise
    if child.returncode != 0:
        raise DeobfuscationFailed(err.decode('utf-8', 'replace'))
    return out.decode('utf-8')


def _normalize_error(stderr: str) -> str:
    match = _ERROR_RE.search(stderr)
    if match is not None:
        return match.group(1)
    return 'ERROR'


def _as_global_script(source: str, calls: tuple[str, ...]) -> str:
    """
    Wrap *source* so it runs under the *script* execution model and then invoke each name in *calls* as
    a host would.

    `node file.js` runs a CommonJS module, where a top-level `var`/`function` is scoped to the module,
    so `behavior` alone cannot observe anything about the global object. Indirect `eval` runs its
    argument in the true global scope — the same scope a browser `<script>`, a Windows Script Host
    `.js`, or a JXA file gets — which is what puts a top-level declaration onto `globalThis` and makes
    it reachable by name from outside.

    That reachability is the whole point: a host entrypoint has no caller inside the file, so its
    deletion changes nothing about running the file and is invisible to a stdout comparison. Calling it
    afterwards through `globalThis` is what turns the deletion into an observable difference, with Node
    rather than our own reading of the specification deciding what the difference is.
    """
    lines = [
        'const globalEval = eval;',
        F'globalEval({json.dumps(source)});',
    ]
    for name in calls:
        key = json.dumps(name)
        lines.append(
            F'console.log({key} + "=" + JSON.stringify('
            F'typeof globalThis[{key}] === "function" ? globalThis[{key}]() : undefined));'
        )
    return '\n'.join(lines)


def host_behavior(
    source: str,
    *,
    calls: tuple[str, ...] = (),
    timeout: float = 15.0,
) -> tuple[str, str | None]:
    """
    The observable behavior of *source* run as a classic global script, including the results of the
    host calling each name in *calls* once loading finishes. Same return shape as `behavior`.
    """
    return behavior(_as_global_script(source, calls), timeout=timeout)


def behavior(source: str, *, timeout: float = 15.0) -> tuple[str, str | None]:
    """
    Execute *source* in Node.js and return its observable behavior as a pair: the captured standard
    output, and the error type (`TypeError`, `ReferenceError`, …) when execution terminated with an
    uncaught exception, or `None` on success. Only the type is kept, not the message: the message
    describes the offending expression, which a semantics-preserving rewrite may legitimately reshape
    (e.g. folding `(function(){})(x)` to `void 0` turns "(intermediate value) is not a function" into
    "(void 0) is not a function" — the same `TypeError`). Stack traces and file paths are dropped too,
    so an original snippet and its deobfuscation compare equal whenever they throw the same way.

    The snippet reaches the file untranslated, because which characters end a line is a question
    the engine is asked here: the platform default rewrites every `\\n` on the way out, so a case
    written with `\\r\\n` handed Node a `\\r\\r\\n`, and the two sides of the comparison then read
    different programs.
    """
    node = node_executable()
    if node is None:
        raise RuntimeError('node.js is not available')
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'snippet.js')
        with open(path, 'w', encoding='utf-8', newline='') as stream:
            stream.write(source)
        proc = subprocess.run(
            [node, path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
        )
    error = None if proc.returncode == 0 else _normalize_error(proc.stderr)
    return proc.stdout, error


_CODE_UNIT_ENCODER = R'''
function enc(x) {
  if (x === undefined) return 'undefined';
  if (x === null) return 'null';
  if (typeof x === 'number') return Object.is(x, -0) ? '-0' : String(x);
  if (typeof x === 'boolean') return String(x);
  if (typeof x === 'function') return 'function';
  if (typeof x === 'symbol') return 'symbol';
  if (typeof x === 'string')
    return 'S[' + Array.from({length: x.length}, function (_, i) {
      return x.charCodeAt(i);
    }).join(',') + ']';
  if (Array.isArray(x)) return 'A[' + x.map(enc).join(',') + ']';
  return 'O' + Object.prototype.toString.call(x);
}
'''


def code_units(expressions: Sequence[str], *, timeout: float = 15.0) -> list[str]:
    """
    Node's value for each expression in *expressions*, rendered as the structure its UTF-16 code
    units give it: a string as `S[...codes...]`, an array as `A[...elements...]`, and everything
    else as a word naming it. The rendering is what makes a pinned value independent of how the
    value is spelled — `'\\uD83D'`, a literal lone high surrogate, and a `String.fromCharCode` call
    all render as `S[55357]` — so a test comparing two of these compares values and not escapes.

    All of *expressions* are evaluated in one process, because a table of them otherwise costs one
    Node start per row. They are therefore evaluated in one scope and in order, which is a scope any
    of them can write to; keep them free of effects.
    """
    probes = '\n'.join(F'console.log(enc({expression}));' for expression in expressions)
    stdout, error = behavior(F'{_CODE_UNIT_ENCODER}\n{probes}\n', timeout=timeout)
    if error is not None:
        raise AssertionError(F'node refused one of {len(expressions)} expressions: {error}')
    return stdout.splitlines()
