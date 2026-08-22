from __future__ import annotations

import unittest

from test.lib.scripts.js.analysis.differential import node_executable
from test.lib.scripts.js.deobfuscation import TestJsDeobfuscator
from test.lib.scripts.js.ledger import before_and_after, each_program_still_prints

from refinery.lib.scripts.js.deobfuscation.protospelling import JsPrototypeSpellingNormalization

#: Expressions that reach an intrinsic prototype without naming it, mapped to the statement each is
#: written back out as. The three object spellings name `Object.prototype` and the array one names
#: `Array.prototype`, which is the prototype an array literal actually inherits.
A_PROTOTYPE_SPELLING_AND_WHAT_NAMES_IT = {
    '({}).__proto__.z = 9;':
        'Object.prototype.z = 9;',
    'Object.getPrototypeOf({}).z = 9;':
        'Object.prototype.z = 9;',
    '({}).constructor.prototype.z = 9;':
        'Object.prototype.z = 9;',
    '[].__proto__.z = 9;':
        'Array.prototype.z = 9;',
    'Object.getPrototypeOf([]).z = 9;':
        'Array.prototype.z = 9;',
    '[].constructor.prototype.z = 9;':
        'Array.prototype.z = 9;',
}

#: Programs whose spelling reaches something other than the intrinsic prototype, mapped to what
#: Node prints for each. Three replace the mechanism the spelling goes through — the `constructor`
#: a receiver inherits, `Object.getPrototypeOf`, and the `__proto__` accessor — and one binds the
#: name `Object` to an object of its own. Four more give an object literal a `__proto__` of its own,
#: which every spelling of that key does: written with a colon it sets the prototype, and written
#: as a shorthand or a computed key it installs an own property that shadows the accessor.
A_SPELLING_THAT_REACHES_SOMETHING_ELSE = {
    'Object.getPrototypeOf = function () { return {q: 5}; };'
    ' console.log(Object.getPrototypeOf({}).q);':
        '5\n',
    "Object.defineProperty(Object.prototype, '__proto__', {get: function () { return {q: 5}; }});"
    ' console.log(({}).__proto__.q);':
        '5\n',
    'var Object = {prototype: {q: 1}, getPrototypeOf: function () { return {q: 5}; }};'
    ' console.log(Object.getPrototypeOf({}).q);':
        '5\n',
    'console.log(({__proto__: {q: 5}}).__proto__.q);':
        '5\n',
    "console.log(({'__proto__': {q: 5}}).__proto__.q);":
        '5\n',
    "console.log(({['__proto__']: {q: 5}}).__proto__.q);":
        '5\n',
    'var __proto__ = {q: 5}; console.log(({__proto__}).__proto__.q);':
        '5\n',
    'console.log(({...{a: 1}}).__proto__ === Object.prototype);':
        'true\n',
}


class TestAPrototypeSpellingIsWrittenAsTheNameItReaches(TestJsDeobfuscator):
    """
    The pass alone, with nothing else in front of it, so that what it rewrites is its own answer
    rather than one another pass reached first.
    """

    def test_each_spelling_is_written_as_the_prototype_it_names(self):
        self.assertEqual(
            {
                source: self._run_transformer(source, JsPrototypeSpellingNormalization)
                for source in A_PROTOTYPE_SPELLING_AND_WHAT_NAMES_IT
            },
            A_PROTOTYPE_SPELLING_AND_WHAT_NAMES_IT,
        )

    def test_a_receiver_the_syntax_does_not_decide_is_left_alone(self):
        """
        A receiver held in a binding reaches the same prototype, and reaching it needs the value the
        binding holds rather than the syntax at the write. It is left as it stands until it does.
        """
        source = 'var a = {};\na.__proto__.z = 9;'
        self.assertEqual(self._run_transformer(source, JsPrototypeSpellingNormalization), source)

    def test_a_name_the_file_binds_is_not_the_intrinsic(self):
        source = 'var Object = { prototype: {} };\nObject.getPrototypeOf({}).z = 9;'
        self.assertEqual(self._run_transformer(source, JsPrototypeSpellingNormalization), source)


@unittest.skipIf(node_executable() is None, 'node.js is not available')
class TestASpellingThatReachesSomethingElseIsNotRewritten(TestJsDeobfuscator):
    """
    The identity each rewrite rests on is the mechanism the language installs, so a file that
    replaces that mechanism means something else by the same spelling. Every answer below was
    established by running Node.
    """

    def test_a_spelling_that_reaches_something_else_still_answers_it(self):
        rows = A_SPELLING_THAT_REACHES_SOMETHING_ELSE
        self.assertEqual(
            {source: before_and_after(source) for source in rows},
            each_program_still_prints(rows),
        )

    def test_a_reassigned_constructor_is_not_the_intrinsic(self):
        """
        Node throws a `ReferenceError` for this program, because `C` is a function expression the
        assignment names nothing outside itself. What the rewrite must not do is answer the read
        from `Object.prototype`, which is the one thing that would make it stop throwing.
        """
        source = (
            'Object.prototype.constructor = function C() { };'
            ' C.prototype.q = 5;'
            ' console.log(({}).constructor.prototype.q);'
        )
        self.assertEqual(before_and_after(source), (('', 'ReferenceError'), ('', 'ReferenceError')))
