"""
Inline constant variable references in PowerShell scripts.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator

from refinery.lib.scripts import (
    Expression,
    Node,
    Transformer,
    _clone_node,
)
from refinery.lib.scripts.ps1.analysis.cache import model_cache
from refinery.lib.scripts.ps1.analysis.dataflow import Ps1VariableFlow
from refinery.lib.scripts.ps1.analysis.model import (
    Binding,
    binding_key,
    is_assignment_write_target,
    is_substitutable_position,
    is_write_occurrence,
)
from refinery.lib.scripts.ps1.analysis.separator import coerced_text_at
from refinery.lib.scripts.ps1.analysis.values import (
    UNKNOWN,
    integer_of,
    make_string_literal,
    read,
    survives_being_written,
    unwrap_to_array_literal,
)
from refinery.lib.scripts.ps1.ast import (
    assignment_of,
    assignment_target_variables,
    unwrap_parens,
)
from refinery.lib.scripts.ps1.data import PS1_KNOWN_VARIABLES
from refinery.lib.scripts.ps1.deobfuscation.helpers import (
    is_array_reverse_call,
    iter_variable_mutations,
)
from refinery.lib.scripts.ps1.deobfuscation.removal import Ps1RemovalPlans
from refinery.lib.scripts.ps1.deobfuscation.substitution import substitute, substitute_field
from refinery.lib.scripts.ps1.model import (
    Ps1ArrayExpression,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1BinaryExpression,
    Ps1CastExpression,
    Ps1ClassDefinition,
    Ps1DoLoop,
    Ps1EnumDefinition,
    Ps1ExpandableString,
    Ps1ExpressionStatement,
    Ps1ForLoop,
    Ps1FunctionDefinition,
    Ps1HereString,
    Ps1IfStatement,
    Ps1IndexExpression,
    Ps1ParenExpression,
    Ps1Pipeline,
    Ps1PipelineElement,
    Ps1ScopeModifier,
    Ps1ScriptBlock,
    Ps1StringLiteral,
    Ps1SwitchStatement,
    Ps1TypeExpression,
    Ps1UnaryExpression,
    Ps1Variable,
    Ps1WhileLoop,
)
from refinery.lib.scripts.ps1.synth import Ps1Synthesizer
from refinery.lib.scripts.win32const import DEFAULT_ENVIRONMENT_TEMPLATE

_PS1_DEFAULT_VARIABLES: dict[str, str] = {
    key.lower(): value for key, value in {
        'ConfirmPreference'          : r'High',
        'ConsoleFileName'            : r'',
        'DebugPreference'            : r'SilentlyContinue',
        'ErrorActionPreference'      : r'Continue',
        'InformationPreference'      : r'SilentlyContinue',
        'ProgressPreference'         : r'Continue',
        'PSCommandPath'              : r'',
        'PSCulture'                  : r'en-US',
        'PSEmailServer'              : r'',
        'PSHome'                     : r'C:\Windows\System32\WindowsPowerShell\v1.0',
        'PSScriptRoot'               : r'',
        'PSSessionApplicationName'   : r'wsman',
        'PSSessionConfigurationName' : r'http://schemas.microsoft.com/powershell/Microsoft.PowerShell',
        'PSUICulture'                : r'en-US',
        'ShellID'                    : r'Microsoft.PowerShell',
        'VerbosePreference'          : r'SilentlyContinue',
        'WarningPreference'          : r'Continue',
    }.items()
}

PS1_ENV_CONSTANTS = {
    lower_key: value
    for key, value in DEFAULT_ENVIRONMENT_TEMPLATE.items()
    if not (lower_key := key.lower()).startswith(('path', 'processor'))
    and '{u}' not in value
    and '{h}' not in value
}

_PS1_AUTOMATIC_VARIABLES = frozenset({
    '_',
    'args',
    'error',
    'event',
    'eventargs',
    'eventsubscriber',
    'executioncontext',
    'false',
    'foreach',
    'home',
    'host',
    'input',
    'lastexitcode',
    'matches',
    'myinvocation',
    'nestedpromptlevel',
    'null',
    'ofs',
    'pid',
    'profile',
    'psboundparameters',
    'pscmdlet',
    'pscommandpath',
    'psitem',
    'psscriptroot',
    'psversiontable',
    'pwd',
    'sender',
    'sourceargs',
    'sourceeventargs',
    'stacktrace',
    'switch',
    'this',
    'true',
})

_PS1_SKIP_VARIABLES = (
    _PS1_AUTOMATIC_VARIABLES
    | frozenset(PS1_KNOWN_VARIABLES)
    | frozenset(_PS1_DEFAULT_VARIABLES)
)

#: The names the engine maintains between statements, so what the script last assigned to one is not
#: what it is worth at the next read: `$_` is rebound per pipeline object, `$Matches` at every
#: `-match`, `$LASTEXITCODE` by every native command, and a preference variable is read by the
#: engine itself. No write of one of these establishes a value this pass may carry to a reader.
_PS1_ENGINE_VARIABLES = _PS1_AUTOMATIC_VARIABLES | frozenset(_PS1_DEFAULT_VARIABLES)

_MIN_EXPANSION_BUDGET = 256


def _collect_mutated_variables(root: Node) -> set[str]:
    """
    Return the set of variable keys that are written to anywhere in the AST. This includes
    assignment targets, ForEach loop variables, ++/-- operands, and parameter declarations.
    """
    mutated: set[str] = set()
    for var, _kind, _node in iter_variable_mutations(root):
        key = _candidate_key(var)
        if key is not None:
            mutated.add(key)
    for node in root.walk():
        if isinstance(node, Ps1ExpressionStatement):
            rv = is_array_reverse_call(node)
            if rv is not None:
                key = _candidate_key(rv)
                if key is not None:
                    mutated.add(key)
    return mutated


def _candidate_key(var: Ps1Variable) -> str | None:
    """
    Return the candidate lookup key for a variable, or `None` if it is not
    eligible for constant inlining.
    """
    if var.scope == Ps1ScopeModifier.NONE:
        return var.name.lower()
    if var.scope == Ps1ScopeModifier.ENV:
        return F'env:{var.name.lower()}'
    return None


def _survives_this_position(value: Node, occurrence: Ps1Variable) -> bool:
    """
    Whether writing *value* where *occurrence* stands leaves the program meaning what it did.

    One value does not: a `System.Decimal` whose value is a whole number written to places. 5.1
    folds a constant expression in its parser and a numeral reaching that fold loses those places,
    so putting one where an operator can reach it moves the computation from run time to parse time
    — measured, `$z = 1.0d; $z + 0d` is `1.0` and the `1.0d + 0d` written for it is `1`. Anywhere an
    operator cannot reach it the numeral is read as itself and the substitution is what it was:
    `$z = 1.0d; ,$z` still writes `1.0`.

    See `refinery.lib.scripts.ps1.analysis.values.survives_being_written` for the value half of this
    and `read_operand` for the rule both stand on.
    """
    if survives_being_written(read(value)):
        return True
    return not isinstance(
        _ancestor_past_parens(occurrence), (Ps1BinaryExpression, Ps1UnaryExpression))


def _ancestor_past_parens(node: Node) -> Node | None:
    """
    The first ancestor of *node* that is not a parenthesis, which is what decides whether an
    operator reaches it: a parenthesis does not stop 5.1 folding what it wraps.
    """
    parent = node.parent
    while isinstance(parent, Ps1ParenExpression):
        parent = parent.parent
    return parent


def _constant_value_key(node: Node) -> tuple | None:
    """
    A hashable key for the constant value of a node, or `None` where the node names no value. Two
    nodes with the same key name the same value, which is what the inliner needs in order to know
    that re-assigning a variable to what it already holds changes nothing.

    The value comes from the domain and not from the spelling, so a Char is one — `[char]39` names
    the apostrophe and is a constant this may carry, where a reader that matched literals by their
    node class saw a cast and stopped. `Ps1Fact` is already the key: it carries the type beside the
    payload, so a Char and the one-character String that holds the same character are different
    keys, which is the whole point of asking the domain rather than the node.

    A type literal is not a value the domain names — `[int]` as a value is a `System.RuntimeType` —
    and it is keyed here by the name it writes, because the inliner only ever compares one of these
    against another.

    """
    node = unwrap_parens(node)
    if isinstance(node, Ps1TypeExpression):
        return ('type', node.name)
    fact = read(node)
    return None if fact is UNKNOWN else ('value', fact)


def _get_array_literal(node: Node) -> Ps1ArrayLiteral | None:
    """
    Return the indexable `refinery.lib.scripts.ps1.model.Ps1ArrayLiteral` from either a bare literal
    or `@(...)`.
    """
    if isinstance(node, Expression):
        return unwrap_to_array_literal(node)
    return None


def _clone_constant(node: Node) -> Expression:
    """
    Create a fresh copy of a constant value node without following parent references. This avoids
    the catastrophic cost of `copy.deepcopy` which traverses the entire AST through parents.

    What is copied is the *spelling* and not the value, deliberately: a numeral the source wrote in
    a command argument keeps its own text there — `Write-Host 1.10` prints `1.10` and
    `notepad.exe 0x10` receives `0x10` — so an inliner that spelled the value afresh would change
    what a command is handed. `@(...)` around a bare list is the one thing normalized away, because
    the parenthesis this adds is what the value needs where it lands.
    """
    unwrapped = unwrap_parens(node)
    if isinstance(unwrapped, Ps1ArrayExpression):
        unwrapped = unwrap_to_array_literal(unwrapped) or unwrapped
    if not isinstance(unwrapped, Expression):
        raise TypeError(F'cannot clone {type(unwrapped).__name__}')
    clone = _clone_node(unwrapped)
    if isinstance(clone, Ps1ArrayLiteral) and len(clone.elements) > 1:
        return Ps1ParenExpression(expression=clone)
    return clone


def _interpolated(value: Expression, site: Ps1Variable, flow: Ps1VariableFlow) -> Expression | None:
    """
    What a value contributes where it is interpolated into an expandable string, which is the text
    it renders to and not the way it was written.

    This is the one place where how a value is spelled and what it renders to are different
    questions, and installing the spelling answered the wrong one: measured, `$s = 0xFF; "$s"` is
    the String `255` on 5.1 where the literal written in reads `0xFF`, and `$c = [char]65; "$c"` is
    `A` where the cast reads as itself. `coerced_text_at` is that second question, asked at *site*
    because a collection is separated by `$OFS`; a value it names no text for is left alone rather
    than written down some other way.

    A here-string is refused because a part is not a standalone literal: the synthesizer writes a
    part's characters into the surrounding quotes, so a spelling that carries its own delimiters has
    nowhere to put them.
    """
    text = coerced_text_at(value, site, flow)
    if text is None:
        return None
    literal = make_string_literal(text)
    return literal if isinstance(literal, Ps1StringLiteral) else None


def _walk_outer_scope(root: Node):
    """
    Walk the AST like `root.walk()` but skip the bodies of function, class, and enum definitions.
    The definition node itself is yielded so that it can still be removed or inspected.
    """
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (Ps1FunctionDefinition, Ps1ClassDefinition, Ps1EnumDefinition)):
            continue
        for child in node.children():
            stack.append(child)


def _find_removable_statement(node: Node) -> Node | None:
    """
    Walk upward from an expression node to find the statement-level node that can be removed from
    its parent's body list.
    """
    cursor = node
    while cursor.parent is not None:
        parent = cursor.parent
        if isinstance(parent, Ps1ExpressionStatement):
            cursor = parent
            continue
        if isinstance(parent, Ps1PipelineElement):
            cursor = parent
            continue
        if isinstance(parent, Ps1Pipeline):
            if len(parent.elements) == 1:
                cursor = parent
                continue
        return cursor
    return None


class _ConstantTable:
    """
    The constant value each write of a script establishes, keyed by the identity of the occurrence
    that writes it, and the constants of the names the script never writes at all.

    Two tables because they answer two questions. A write is a point in the program and the flow
    model orders it against a read; an ambient constant is a value the engine established before the
    script ran — `$env:ComSpec`, `$ErrorActionPreference` — and there is no point to order it
    against. The pass this replaces held one table for both and used the absence of a position as
    the marker, so a write it could not place and a value that has no position were the same entry.

    A write with no entry here is not a lesser kind of write. It is a write whose value this pass has
    nothing to say about, and the flow model orders and kills it exactly as it does any other:
    sorting writes by whether their value happens to be constant is what made `if ($c) { $x = 'b' }`
    fold and `if ($c) { $x = $y }` refuse.
    """

    def __init__(self, root: Node):
        self.by_write: dict[int, Expression] = {}
        self.ambient: dict[str, Expression] = {}
        self.values: defaultdict[str, list[Expression]] = defaultdict(list)
        self._collect_writes(root)
        self._collect_ambient(root)

    def _collect_writes(self, root: Node):
        for node in root.walk():
            if not isinstance(node, Ps1AssignmentExpression):
                continue
            if node.operator != '=' or node.value is None:
                continue
            targets = assignment_target_variables(node.target)
            if len(targets) != 1:
                continue
            key = _candidate_key(targets[0])
            if key is None or key in _PS1_ENGINE_VARIABLES:
                # A preference or automatic variable is the engine's as much as the script's: it
                # reads and writes these names between statements — `$_` per pipeline object,
                # `$Matches` at every `-match` — so what the script last assigned is not what the
                # name is worth. The ambient table is the only thing that may answer for one, and it
                # answers only while the script leaves the name alone.
                continue
            if _constant_value_key(node.value) is None:
                continue
            value = unwrap_parens(node.value)
            self.by_write[id(targets[0])] = value
            self.values[key].append(value)

    def _collect_ambient(self, root: Node):
        """
        A default the engine supplies is only this name's value while the script leaves the name
        alone. Any write of it anywhere replaces the default with something this table has no claim
        on — including a write inside a block, which `Ps1SemanticModel` binds locally but `. { }`
        performs on the caller, and a write that reaches *through* the name, which
        `is_write_occurrence` counts as the write it is.

        A write nobody can attribute is *not* collected here, because it is not a fact about the
        whole script: it lands at a point, and an ambient default is a definition at the script's
        entry, so the two are ordered like anything else. `Ps1VariableFlow.ambient_value_survives`
        asks that per read — silencing every default here instead was measured, and it costs the
        `$PSHome` unpacking that an obfuscated loader's first stage is built out of.
        """
        touched: set[str] = set()
        for node in root.walk():
            if isinstance(node, Ps1Variable) and is_write_occurrence(node):
                touched.add(binding_key(node))
        for key, value in _PS1_DEFAULT_VARIABLES.items():
            if key not in touched:
                self._add_ambient(key, make_string_literal(value))
        for name, value in PS1_ENV_CONSTANTS.items():
            key = F'env:{name}'
            if key not in touched:
                self._add_ambient(key, make_string_literal(value))

    def _add_ambient(self, key: str, value: Expression):
        self.ambient[key] = value
        self.values[key].append(value)

    def __bool__(self) -> bool:
        return bool(self.by_write or self.ambient)


class _InlineRecord:
    """
    The read occurrences one substitution walk replaced, per binding.

    The count is what licenses removing the binding's writes — every occurrence in `Binding.reads`
    has to be accounted for, and that set includes reads this pass never walked, such as one inside a
    function body — and the replacement nodes are where the value now stands, so they are what the
    pass can point at when it claims that removing the write destroys nothing.
    """

    def __init__(self):
        self._bindings: dict[int, Binding] = {}
        self._replacements: defaultdict[int, list[Node]] = defaultdict(list)

    def add(self, binding: Binding, replacement: Node):
        self._bindings[id(binding)] = binding
        self._replacements[id(binding)].append(replacement)

    def __iter__(self) -> Iterator[tuple[Binding, list[Node]]]:
        for key, binding in self._bindings.items():
            yield binding, self._replacements[key]


class _Inlining:
    """
    The state one substitution walk carries: the constants it may install, the flow model that says
    which of them a read observes, the keys the expansion budget has already refused, the variable
    occurrences an enclosing index expression has already spoken for, and what was replaced.
    """

    def __init__(self, table: _ConstantTable, flow: Ps1VariableFlow, blocked: set[str]):
        self.table = table
        self.flow = flow
        self.blocked = blocked
        self.handled: set[int] = set()
        self.record = _InlineRecord()

    def value_at(self, var: Ps1Variable, key: str) -> Expression | None:
        """
        The constant *var* holds where it stands, or `None` when no single value does.
        """
        binding = self.binding_of(var)
        if binding is None:
            value = self.table.ambient.get(key)
            if value is None or not self.flow.ambient_value_survives(var):
                return None
            return value
        write = self.flow.reaching_definition(var)
        if write is None:
            return None
        return self.table.by_write.get(id(write))

    def binding_of(self, var: Ps1Variable) -> Binding | None:
        return self.flow.semantic.binding_of(var)

    def installed(self, var: Ps1Variable, replacement: Node):
        binding = self.binding_of(var)
        if binding is not None:
            self.record.add(binding, replacement)


class Ps1ConstantInlining(Transformer):

    def __init__(self, max_expansion_ratio: float = 0.2, min_inlines_to_prune: int | None = 1):
        super().__init__()
        self.max_expansion_ratio = max_expansion_ratio
        self.min_inlines_to_prune = min_inlines_to_prune

    def visit(self, node: Node):
        # Captured once rather than re-read per reference: every substitution below marks the pass
        # changed, which drops the cache, so a per-site lookup would rebuild the control-flow graphs
        # of the whole script once per inlined variable. Nothing this pass adds or removes is a
        # statement, so the graphs it would rebuild are the graphs it already has.
        flow = model_cache(self, node).variable_flow
        table = _ConstantTable(node)
        if not table:
            return None
        state = _Inlining(table, flow, self._blocked_by_expansion(node, table))
        self._substitute(node, state)
        self._remove_dead_assignments(table, state)
        return None

    def _blocked_by_expansion(self, root: Node, table: _ConstantTable) -> set[str]:
        """
        The keys whose substitution would grow the script past the expansion budget, estimated over
        every reference before any of them is installed. Purely a size heuristic: it withholds an
        inlining that is correct, and it is asked before the flow model so that a script full of
        references to one large array does not pay for a reaching-definition query per reference.
        """
        synth = Ps1Synthesizer()
        script_size = len(synth.convert(root))
        max_budget = max(_MIN_EXPANSION_BUDGET, int(script_size * self.max_expansion_ratio))

        value_lengths: dict[str, int] = {}
        array_literals: dict[str, Ps1ArrayLiteral | None] = {}
        for key, values in table.values.items():
            value_lengths[key] = max(len(synth.convert(value)) for value in values)
            array_literals[key] = _get_array_literal(values[0])
        elem_lengths: dict[tuple[str, int], int] = {}

        expansion: defaultdict[str, int] = defaultdict(int)
        for node in _walk_outer_scope(root):
            if isinstance(node, Ps1IndexExpression):
                var = node.object
                if not isinstance(var, Ps1Variable):
                    continue
                key = _candidate_key(var)
                if key is None or key not in table.values or node.index is None:
                    continue
                idx = integer_of(read(node.index))
                if idx is not None:
                    array = array_literals[key]
                    if array is None:
                        continue
                    if not 0 <= idx < len(array.elements):
                        continue
                    ref_len = 1 + len(var.name) + 1 + len(synth.convert(node.index)) + 1
                    cache_key = (key, idx)
                    if cache_key not in elem_lengths:
                        elem_lengths[cache_key] = len(synth.convert(array.elements[idx]))
                    expansion[key] += max(0, elem_lengths[cache_key] - ref_len)
                elif isinstance(table.values[key][0], (Ps1StringLiteral, Ps1HereString)):
                    expansion[key] += max(0, value_lengths[key] - (1 + len(var.name)))
            elif isinstance(node, Ps1Variable):
                key = _candidate_key(node)
                if key is None or key not in table.values or is_write_occurrence(node):
                    continue
                expansion[key] += max(0, value_lengths[key] - (1 + len(node.name)))

        return {key for key in table.values if expansion[key] > max_budget}

    def _substitute(self, root: Node, state: _Inlining):
        """
        Replace every reference this pass can resolve with the value it observes.

        The walk stops at a function, class, or enum body. What a read inside one observes is a
        question about the call sites that reach it, which the flow model refuses rather than
        answers, so descending would only spend a query per reference to be told nothing; and a
        class body is opaque to the graphs, so a property initializer inside one locates to the
        class statement and would be ordered against code it does not run beside.

        Which positions may hold a value at all is
        `refinery.lib.scripts.ps1.analysis.model.is_substitutable_position`, asked once here rather
        than reassembled from the positional predicates it is made of. It is a fact about the
        *position*, not about the binding: a store through is a write occurrence carrying no value,
        so the flow model already names none for a read below one, but the ambient table answers
        with no binding at all and would otherwise install a constant where `$x[0] = 'z'` names a
        place rather than a value.
        """
        for node in list(_walk_outer_scope(root)):
            if isinstance(node, Ps1IndexExpression):
                var = node.object
                if not isinstance(var, Ps1Variable):
                    continue
                # Spoken for either way: the walk snapshot still holds this occurrence after the
                # index expression around it has been swapped out, and substituting it a second time
                # would install the whole value where an element of it now stands.
                state.handled.add(id(var))
                key = _candidate_key(var)
                if key is not None and is_substitutable_position(var):
                    self._substitute_index_reference(node, var, key, state)
            elif isinstance(node, Ps1Variable):
                if id(node) in state.handled or not is_substitutable_position(node):
                    continue
                key = _candidate_key(node)
                if key is not None and key not in state.blocked:
                    self._substitute_variable_reference(node, key, state)

    def _substitute_index_reference(
        self,
        node: Ps1IndexExpression,
        var: Ps1Variable,
        key: str,
        state: _Inlining,
    ) -> None:
        const_value = state.value_at(var, key)
        if const_value is None:
            return
        idx = integer_of(read(node.index))
        if idx is None:
            if key in state.blocked or not isinstance(const_value, Ps1StringLiteral):
                return
            replacement = _clone_constant(const_value)
            if substitute_field(node, 'object', replacement):
                self.mark_changed()
                state.installed(var, replacement)
            return
        if isinstance(const_value, Ps1StringLiteral):
            text = const_value.value
            if not 0 <= idx < len(text):
                return
            replacement = make_string_literal(text[idx])
        else:
            array = _get_array_literal(const_value)
            if array is None or not 0 <= idx < len(array.elements):
                return
            replacement = _clone_constant(array.elements[idx])
        if substitute(node, replacement):
            self.mark_changed()
            state.installed(var, replacement)

    def _substitute_variable_reference(
        self,
        node: Ps1Variable,
        key: str,
        state: _Inlining,
    ) -> None:
        const_value = state.value_at(node, key)
        if const_value is None or not _survives_this_position(const_value, node):
            return
        if isinstance(node.parent, Ps1ExpandableString):
            replacement = _interpolated(const_value, node, state.flow)
        else:
            replacement = _clone_constant(const_value)
        if replacement is None:
            return
        if substitute(node, replacement):
            self.mark_changed()
            state.installed(node, replacement)

    def _remove_dead_assignments(self, table: _ConstantTable, state: _Inlining):
        """
        Delete the constant writes of every binding whose value nothing observes any more.

        Removal is decided per binding, and counted against that binding's own reads rather than
        against the references this walk resolved. Every occurrence in `Binding.reads` observes the
        value, including the ones the walk cannot answer for — a read inside a function body, a read
        after a write whose value is not constant, an index this pass cannot evaluate — and each of
        them is a reader the write still has. Counting only what the walk replaced is what let
        `$x = 'a'; function f { Write-Host $x }; Write-Host $x; f` delete the assignment `f` reads.

        A write that observes the previous value is a read as much as a write, so a binding with one
        is never dead however many of its reads were substituted; and a write whose value this pass
        holds no constant for stays, because deleting it would drop whatever it does to produce that
        value.

        `Binding.reads` is the whole list of readers only when the binding ends with its own body.
        `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel` binds a bare write to the
        block it is written in, and `refinery.lib.scripts.ps1.analysis.blocks.Ps1BlockModel` is
        the layer that says a `. { }` or a `ForEach-Object` body performs that write on whoever runs
        it — where the readers are other bindings entirely, and `_block_kills` already honours the
        same fact on the read side. Deleting such a write leaves the caller reading the value from
        before the body.
        """
        plans = Ps1RemovalPlans()
        for binding, replacements in state.record:
            if len(replacements) < len(binding.reads):
                continue
            if self._writes_leave_the_body(state.flow, binding):
                continue
            if any(write.role.observes for write in binding.writes):
                continue
            if self.min_inlines_to_prune is not None:
                if len(replacements) < self.min_inlines_to_prune:
                    continue
            for write in binding.writes:
                assignment = assignment_of(write.node)
                if assignment is None or id(write.node) not in table.by_write:
                    continue
                statement = self._find_removable_statement(assignment)
                if statement is not None:
                    plans.propose(statement)
        if plans.commit():
            self.mark_changed()

    @staticmethod
    def _writes_leave_the_body(flow: Ps1VariableFlow, binding: Binding) -> bool:
        """
        Whether *binding*'s writes land in the scope of whatever runs the body they are written in,
        rather than in a scope that ends with that body. True for every block but a proven child
        scope — see `refinery.lib.scripts.ps1.analysis.blocks` for why that asymmetry is the safe
        one.
        """
        node = binding.scope.node
        return isinstance(node, Ps1ScriptBlock) and flow.blocks.may_write_caller_scope(node)

    _find_removable_statement = staticmethod(_find_removable_statement)


class Ps1NullVariableInlining(Transformer):
    """
    Replace references to never-assigned variables with `$Null`. Only operates on variables that
    appear in expression contexts where null coercion enables further simplification (arithmetic,
    comparison, cast, assignment value).
    """

    @staticmethod
    def _is_null_eligible(ref: Ps1Variable) -> bool:
        cursor = ref
        while cursor.parent is not None:
            parent = cursor.parent
            if isinstance(parent, Ps1BinaryExpression):
                return True
            if isinstance(parent, Ps1UnaryExpression):
                return True
            if isinstance(parent, Ps1CastExpression):
                cursor = parent
                continue
            if isinstance(parent, Ps1AssignmentExpression) and cursor is parent.value:
                return True
            if isinstance(parent, (Ps1ParenExpression, Ps1ArrayLiteral)):
                cursor = parent
                continue
            if isinstance(parent, (Ps1WhileLoop, Ps1DoLoop, Ps1ForLoop)) and cursor is parent.condition:
                return True
            if isinstance(parent, (Ps1IfStatement, Ps1SwitchStatement)):
                return any(cursor is cond for cond, _ in parent.clauses)
            return False
        return False

    def visit(self, node: Node):
        mutated = _collect_mutated_variables(node)
        for ref in list(node.walk()):
            if not isinstance(ref, Ps1Variable):
                continue
            key = _candidate_key(ref)
            if key is None:
                continue
            if key in mutated:
                continue
            if key in PS1_KNOWN_VARIABLES:
                continue
            if key in _PS1_DEFAULT_VARIABLES:
                continue
            if key in _PS1_AUTOMATIC_VARIABLES:
                continue
            if key.startswith('env:'):
                continue
            if is_assignment_write_target(ref):
                continue
            if not self._is_null_eligible(ref):
                continue
            if not substitute(ref, Ps1Variable(name='Null')):
                continue
            self.mark_changed()
