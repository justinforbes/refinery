"""
Rewrite the spellings that reach an intrinsic prototype without naming it to `Owner.prototype`.

A prototype pollution gadget writes `Object.prototype` through a literal rather than through the
name — `({}).__proto__.z = 9`, `Object.getPrototypeOf({}).z = 9`,
`({}).constructor.prototype.z = 9` — and every one of them means the same object as
`Object.prototype`. Spelling it that way instead is what lets the rest of the pipeline see the
write at all: a write is attributed to the name at the root of the member chain it is written
through, so a chain rooted in a literal is attributed to nothing and every question about that
prototype goes on being answered from a table of what the language puts there.

The rewrite is the same shape as
`refinery.lib.scripts.js.deobfuscation.globalfinder.JsGlobalFinderInlining`, which already rewrites
a call that computes the global object to `globalThis`, and it earns its place the same way twice
over: the file the analyst reads says which object is being patched, and every existing check sees
the write with no new analysis behind it.

The identity is not unconditional and each gate below was established by running Node. It holds
because the mechanism each spelling goes through is the one the language installs, so a file that
replaces that mechanism is asking for something else — `Object.prototype.constructor = C` makes
`({}).constructor.prototype` mean `C.prototype`, and replacing the `__proto__` accessor or
`Object.getPrototypeOf` redirects the other two. Those are asked of the effect model per key,
since asking per name would refuse every file this exists for: a file that writes
`Object.prototype.z` has written `Object`.
"""
from __future__ import annotations

from refinery.lib.scripts import Node, _replace_in_parent
from refinery.lib.scripts.js.analysis.cache import model_cache
from refinery.lib.scripts.js.analysis.effects import EffectModel
from refinery.lib.scripts.js.analysis.model import SemanticModel
from refinery.lib.scripts.js.deobfuscation.helpers import ScriptLevelTransformer, access_key
from refinery.lib.scripts.js.model import (
    JsArrayExpression,
    JsCallExpression,
    JsIdentifier,
    JsMemberExpression,
    JsObjectExpression,
    JsScript,
    strip_parens,
)

_PROTO_KEY = '__proto__'

_GET_PROTOTYPE_OF = 'getPrototypeOf'

_CONSTRUCTOR_KEY = 'constructor'


class JsPrototypeSpellingNormalization(ScriptLevelTransformer):
    """
    Replace each expression that reaches an intrinsic prototype through a literal with the
    `Owner.prototype` that names it. See the module documentation for the spellings recognized and
    the gates the identity rests on.
    """

    self_converging = True

    def _process_script(self, node: JsScript) -> None:
        cache = model_cache(self, node)
        for candidate in list(node.walk()):
            owner = self._owner_whose_prototype_is_named(candidate, cache.model, cache.effects)
            if owner is None:
                continue
            _replace_in_parent(candidate, _a_prototype_of(owner))
            self.mark_changed()

    def _owner_whose_prototype_is_named(
        self,
        node: Node,
        model: SemanticModel,
        effects: EffectModel,
    ) -> str | None:
        """
        The name of the intrinsic whose `prototype` *node* denotes, or `None` where *node* is not
        one of the recognized spellings or a gate refuses it.
        """
        if isinstance(node, JsMemberExpression):
            key = access_key(node)
            if key == _PROTO_KEY:
                return self._owner_read_through_proto(node, model, effects)
            if key == 'prototype':
                return self._owner_read_through_constructor(node, model, effects)
            return None
        if isinstance(node, JsCallExpression):
            return self._owner_read_through_get_prototype_of(node, model, effects)
        return None

    def _owner_read_through_proto(
        self,
        node: JsMemberExpression,
        model: SemanticModel,
        effects: EffectModel,
    ) -> str | None:
        """
        The owner of `<literal>.__proto__`. The accessor that answers it is installed on
        `Object.prototype` whatever the receiver is, so replacing it there redirects the spelling
        for an array literal as much as for an object one.
        """
        owner = _owner_of_literal(node.object)
        if owner is None or effects.global_key_written('Object', _PROTO_KEY):
            return None
        return owner if _names_the_intrinsic(owner, node, model) else None

    def _owner_read_through_constructor(
        self,
        node: JsMemberExpression,
        model: SemanticModel,
        effects: EffectModel,
    ) -> str | None:
        """
        The owner of `<literal>.constructor.prototype`. The `constructor` the receiver inherits is a
        data property of the owner's own prototype, so a file that writes that key is naming its own
        function rather than the intrinsic.
        """
        inner = strip_parens(node.object)
        if not isinstance(inner, JsMemberExpression) or access_key(inner) != _CONSTRUCTOR_KEY:
            return None
        owner = _owner_of_literal(inner.object)
        if owner is None or effects.global_key_written(owner, _CONSTRUCTOR_KEY):
            return None
        return owner if _names_the_intrinsic(owner, node, model) else None

    def _owner_read_through_get_prototype_of(
        self,
        node: JsCallExpression,
        model: SemanticModel,
        effects: EffectModel,
    ) -> str | None:
        """
        The owner of `Object.getPrototypeOf(<literal>)`, where `Object` is the intrinsic and the
        method is the one it was given.
        """
        callee = strip_parens(node.callee)
        if not isinstance(callee, JsMemberExpression) or access_key(callee) != _GET_PROTOTYPE_OF:
            return None
        base = strip_parens(callee.object)
        if not isinstance(base, JsIdentifier) or base.name != 'Object':
            return None
        if len(node.arguments) != 1:
            return None
        owner = _owner_of_literal(node.arguments[0])
        if owner is None or effects.global_key_written('Object', _GET_PROTOTYPE_OF):
            return None
        if not _names_the_intrinsic('Object', node, model):
            return None
        return owner if _names_the_intrinsic(owner, node, model) else None


def _owner_of_literal(node: Node | None) -> str | None:
    """
    The intrinsic whose prototype a literal receiver inherits, or `None` for a receiver whose
    prototype is not knowable from the syntax alone.

    An array literal always inherits `Array.prototype`: no spelling of one says otherwise, an
    elision and a spread included. An object literal is taken only when it is empty, because every
    way of writing a `__proto__` key changes what the receiver answers — written with a colon it
    sets the prototype, and written as a shorthand or a computed key it installs an own property
    that shadows the accessor — and a spread carries one in from a value the syntax does not show.
    """
    node = strip_parens(node)
    if isinstance(node, JsArrayExpression):
        return 'Array'
    if isinstance(node, JsObjectExpression) and not node.properties:
        return 'Object'
    return None


def _names_the_intrinsic(name: str, at: Node, model: SemanticModel) -> bool:
    """
    Whether *name* denotes the intrinsic at *at* rather than something the file bound. A file
    declaring `var Object = {prototype: {}}` names its own object with it, and rewriting a spelling
    to `Object.prototype` there would name that one instead of the prototype the spelling reaches.
    """
    scope = model.scope_of(at)
    return scope is not None and model.lookup(name, scope) is None


def _a_prototype_of(owner: str) -> JsMemberExpression:
    return JsMemberExpression(
        object=JsIdentifier(name=owner),
        property=JsIdentifier(name='prototype'),
        computed=False,
    )
