"""
The effect layer of the PowerShell analysis substrate: whether evaluating a node produces an
observable side effect, and what a standalone statement contributes to the body it sits in. Every
pass that decides "is it safe to delete this?" asks here, so that no two of them can disagree.

These are free functions rather than a model class because the facts they compute are syntactic: a
conservative allow-list over one expression, needing no information from anywhere else in the tree.
A cached model arrives with the first genuine summary fact — interprocedural purity, which has to be
computed over the `refinery.lib.scripts.ps1.analysis.model.Ps1SemanticModel`.

**Scope.** Three questions about a statement are separable, and only two are answered here. Whether
it performs a side effect is `statement_effect`; whether it can put a value on its body's output is
`output_is_covered`, which is not the same question — `Write-Host x` acts and emits nothing. Whether
it may *throw* is the third and has no representation at all: `StatementEffect` has no member for
it, so the trap and try/catch passes keep statement predicates of their own, and folding those in
requires deciding a fault semantics first. It is not a simplification that can be made silently.

One site still answers an emission question with a purity verdict: `_command_body_is_pure` reads an
`EFFECT` statement as disqualifying, where what it means to ask is whether the body emits. It costs
recall rather than safety — a body that acts is kept — and it is named here so the split above is
read as incomplete rather than as done.
"""
from __future__ import annotations

import enum

from typing import Iterator, Sequence, TypeGuard

from refinery.lib.scripts import Block, Node
from refinery.lib.scripts.ps1 import data
from refinery.lib.scripts.ps1.analysis.constants import is_truthy
from refinery.lib.scripts.ps1.analysis.types import TypeOracle
from refinery.lib.scripts.ps1.ast import (
    extract_new_object,
    get_body,
    get_command_name,
    get_member_name,
    get_named_blocks,
    get_param_block,
    is_builtin_variable,
    normalize_dotnet_type_name,
    resolve_command_name,
)
from refinery.lib.scripts.ps1.model import (
    Expression,
    Ps1AccessKind,
    Ps1ArrayExpression,
    Ps1ArrayLiteral,
    Ps1AssignmentExpression,
    Ps1Attribute,
    Ps1BinaryExpression,
    Ps1CastExpression,
    Ps1ClassDefinition,
    Ps1CommandArgument,
    Ps1CommandInvocation,
    Ps1DataSection,
    Ps1DoLoop,
    Ps1EnumDefinition,
    Ps1Exit,
    Ps1ExpandableString,
    Ps1ExpressionStatement,
    Ps1FileRedirection,
    Ps1ForEachLoop,
    Ps1ForLoop,
    Ps1FunctionDefinition,
    Ps1HashLiteral,
    Ps1HereString,
    Ps1IfStatement,
    Ps1IndexExpression,
    Ps1IntegerLiteral,
    Ps1InvokeMember,
    Ps1Jump,
    Ps1MemberAccess,
    Ps1MergingRedirection,
    Ps1ParamBlock,
    Ps1ParenExpression,
    Ps1Pipeline,
    Ps1PipelineElement,
    Ps1RangeExpression,
    Ps1RealLiteral,
    Ps1RedirectionStream,
    Ps1ReturnStatement,
    Ps1Script,
    Ps1ScriptBlock,
    Ps1StringLiteral,
    Ps1SubExpression,
    Ps1SwitchStatement,
    Ps1TrapStatement,
    Ps1TryCatchFinally,
    Ps1TypeExpression,
    Ps1UnaryExpression,
    Ps1Variable,
    Ps1WhileLoop,
)


def _canonical_type_name(name: str) -> str:
    """
    Resolve a purity allow-list type spelling to the lowercased canonical .NET `FullName` the effect
    checks key on, raising when the collected metadata carries no such type. Building the tables
    through this at import time is the fail-loud floor the metadata rework exists to provide: an
    entry that names a type the current data cannot resolve stops the module from loading rather
    than going silently unmatched, which is how a stale allow-list used to fail open. A generic
    type is named by its arity-marked definition (`collections.generic.list` `` `1 ``), the only
    spelling `refinery.lib.scripts.ps1.data.resolve_type` resolves without its type arguments.
    """
    resolved = data.resolve_type(name)
    if resolved is None:
        raise ValueError(
            F'the PowerShell purity allow-list names {name!r}, which the collected metadata does '
            F'not resolve to a type; the data and the allow-list are out of step.'
        )
    return resolved.lower()


def _canonical_type_set(names: set[str]) -> frozenset[str]:
    """
    A frozenset of canonical type keys built from readable source spellings through
    `_canonical_type_name`. Spellings that name the same type collapse to one entry, retiring the
    dual-spelling entries (`int` beside `int32`) the allow-lists carried before the data could
    resolve them.
    """
    return frozenset(_canonical_type_name(name) for name in names)


def _canonical_method_set(entries: set[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    """
    A frozenset of `(canonical type key, lowercased member)` pairs, the form the static-method
    checks look up. Only the type half is resolved through the data and floored by it; the member
    name is matched against a `refinery.lib.scripts.ps1.model.Ps1InvokeMember.member` at its own
    casing.
    """
    return frozenset(
        (_canonical_type_name(type_name), member.lower())
        for type_name, member in entries
    )


def _canonical_read_set(entries: set[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    """
    A frozenset of `(canonical type key, lowercased member)` reads, each floored against the data the
    way `_canonical_type_name` floors a type: the type must resolve, and the member must be collected
    as a reflection property or field. An *instance* field needs no entry — reading a bare memory slot
    runs no code — so this table is for a property (whose getter may run code) or a *static* field
    (whose first read runs the declaring type's static constructor). An entry that names an Extended
    Type System member (which runs code) or a member the type no longer carries fails the load rather
    than silently granting a read the data cannot vouch for. It keys on the same canonical `FullName`
    as the invoke-side tables, so the per-member effect data a later regeneration adds can replace it
    without rekeying.
    """
    result: set[tuple[str, str]] = set()
    for type_name, member in entries:
        type_key = _canonical_type_name(type_name)
        record = data.member_record(type_key, member)
        gated = isinstance(record, dict) and record['source'] == 'reflection' and (
            record['kind'] == 'property'
            or (record['kind'] == 'field' and record.get('static') is True)
        )
        if not gated:
            raise ValueError(
                F'the PowerShell pure-read allow-list names {type_name}.{member}, which the '
                F'collected metadata does not carry as a reflection property or static field; an '
                F'instance field is pure without listing, so the data and allow-list are out of step.'
            )
        result.add((type_key, member.lower()))
    return frozenset(result)


def _canonical_command_set(names: set[str]) -> frozenset[str]:
    """
    A frozenset of lowercased command names, each floored against the collected metadata the way
    `_canonical_type_name` floors a type: a name no capture host reported is a name this module has
    no evidence about, and a table that keeps it fails open the moment the command is renamed or the
    module holding it stops shipping. The load fails instead.

    This is for a table keyed on what a command *is*, never for a deny-list of what a script may do:
    `refinery.lib.scripts.ps1.analysis.world` names `start-threadjob` and `remove-alias`, which the
    capture host does not carry, and a deny-list has to be allowed to outrun the data.
    """
    result: set[str] = set()
    for name in names:
        if data.command(name) is None:
            raise ValueError(
                F'the PowerShell command table names {name!r}, which the collected metadata does '
                F'not carry as a command; the data and the table are out of step.'
            )
        result.add(name.lower())
    return frozenset(result)


def _canonical_sealed_value_type_set(names: set[str]) -> frozenset[str]:
    """
    A frozenset of canonical type keys like `_canonical_type_set`, additionally floored against the
    shipped `sealed` flag. A type on the pure-read allow-list must be sealed: the whole-surface
    grant bets that a value of the type carries exactly the members reflection reports, which a
    subtype could violate. An entry the collected metadata does not mark sealed fails the load,
    retiring the sealedness the table used to assert by hand.
    """
    keys = _canonical_type_set(names)
    for key in keys:
        if not data.type_is_sealed(key):
            raise ValueError(
                F'the PowerShell pure-read allow-list names {key!r}, which the collected metadata '
                F'does not mark sealed; a subtype could carry a member the whole-surface grant does '
                F'not see, so the data and the allow-list are out of step.'
            )
    return keys


#: Types whose entire static surface is granted purity at once. Listing a type here asserts that no
#: static member of it writes anything — a bet re-checked against the real .NET surface whenever an
#: entry is added, because a single writing member makes every call on the type removable.
#: `_IMPURE_STATIC_METHODS` and `_MUTATING_STATIC_METHODS` carve out the members where the bet is
#: wrong but the type is still worth granting wholesale, and out-parameters are handled generically,
#: from the collected signature in `_writes_through_out_parameter` and syntactically in
#: `_is_writable_reference`, rather than per method. The readable source spellings are resolved to
#: canonical `FullName` keys at import, so a spelling variant is one entry and an unresolvable name
#: fails the load.
_PURE_STATIC_METHOD_TYPES = _canonical_type_set({
    'bitconverter',
    'char',
    'collections.arraylist',
    'collections.generic.dictionary`2',
    'collections.generic.hashset`1',
    'collections.generic.list`1',
    'convert',
    'datetime',
    'decimal',
    'double',
    'guid',
    'hashtable',
    'int',
    'int64',
    'io.path',
    'ipaddress',
    'math',
    'object',
    'securestring',
    'string',
    'text.stringbuilder',
    'timespan',
    'version',
})

_PURE_STATIC_METHODS = _canonical_method_set({
    ('diagnostics.process', 'getcurrentprocess'),
    ('threading.tasks.task', 'delay'),
    ('array', 'asreadonly'),
    ('array', 'binarysearch'),
    ('array', 'createinstance'),
    ('array', 'empty'),
    ('array', 'indexof'),
    ('array', 'lastindexof'),
    ('environment', 'expandenvironmentvariables'),
    ('environment', 'getcommandlineargs'),
    ('environment', 'getenvironmentvariable'),
    ('environment', 'getenvironmentvariables'),
    ('environment', 'getfolderpath'),
    ('environment', 'getlogicaldrives'),
})

#: Static members that mutate an argument in place — `[Array]::Reverse($buffer)` rewrites the array
#: it is handed — on a type whose remaining static surface is pure enough to keep granting
#: wholesale. The mutation is invisible to the signature: the argument is passed by value, and an
#: array is a reference the call writes through without an `out`/`byref` marker, so these cannot
#: fold into `_writes_through_out_parameter` and stay a hand-kept table. Deleting the table would
#: let `[Convert]::ToBase64CharArray(...)`, which writes its output array the same unmarked way,
#: reach the `convert` whole-type grant and be removed. Each is pure only when handed a temporary
#: nothing else can read; that is `_denotes_shared_storage`.
_MUTATING_STATIC_METHODS = _canonical_method_set({
    ('array', 'clear'),
    ('array', 'constrainedcopy'),
    ('array', 'copy'),
    ('array', 'fill'),
    ('array', 'reverse'),
    ('array', 'setvalue'),
    ('array', 'sort'),
    ('convert', 'tobase64chararray'),
})

#: Members that do something observable whatever they are handed, on a type whose remaining static
#: surface is pure enough to keep granting wholesale. Unlike `_MUTATING_STATIC_METHODS` these are
#: not saved by being called on a temporary: `[IO.Path]::GetTempFileName()` takes no arguments and
#: still creates a file on disk.
_IMPURE_STATIC_METHODS = _canonical_method_set({
    ('io.path', 'gettempfilename'),
})

#: Types whose entire reflection-read surface is pure: the sealed, immutable value types whose
#: property and field getters only return stored data and whose members never mutate or throw. A
#: read of any member of one of these — present or absent — has no side effect, which is what lets a
#: member the type does not carry resolve to `$null` safely. The bet is the type is sealed, so the
#: runtime value is exactly this type and not a subtype with a member of its own; the shipped
#: `sealed` flag is checked at import by `_canonical_sealed_value_type_set`, so an entry the data
#: does not mark sealed fails the load rather than resting on a hand assertion. Absent-safe stays
#: scoped to this set rather than to any sealed type: a value of one of these carries no member a
#: module's `types.ps1xml` could have added that our capture missed, which sealedness alone does
#: not guarantee for a reference type. `IPAddress` is deliberately absent — its `.Address` throws
#: for an IPv6 address and its `.ScopeId` for an IPv4 one, so its read surface is not uniformly
#: pure. A value type that ever gained an effectful reflection getter would move to `_PURE_READS`,
#: member by member.
_PURE_READ_TYPES = _canonical_sealed_value_type_set({
    'byte',
    'datetime',
    'decimal',
    'double',
    'guid',
    'int',
    'int16',
    'int64',
    'sbyte',
    'single',
    'string',
    'timespan',
    'uint16',
    'uint32',
    'uint64',
    'version',
})

#: Individual reflection reads granted purity where the type's surface as a whole is not. Each
#: `(type, member)` asserts that reading it runs no observable code and cannot throw — a property
#: whose getter only returns cached data (`Process.ProcessName`, where `Process.ExitCode` throws
#: until exit and is deliberately absent), or a static field whose declaring type's static
#: constructor is inert (`Math.PI`). The floor confirms each names a collected reflection property
#: or field, so an entry that turns into an Extended Type System member across a regeneration fails
#: the load rather than vouching for a member that now runs code.
_PURE_READS = _canonical_read_set({
    ('diagnostics.process', 'processname'),
    ('environment', 'machinename'),
    ('environment', 'systemdirectory'),
    ('environment', 'username'),
    ('math', 'e'),
    ('math', 'pi'),
    ('threading.tasks.task', 'status'),
    ('threading.thread', 'currentthread'),
    ('threading.thread', 'managedthreadid'),
})

#: Type names that denote a by-reference wrapper. `[Ref]` is the PowerShell shorthand; the framework
#: name it resolves to spells the same thing and appears in obfuscated scripts.
_REFERENCE_TYPE_NAMES = frozenset({
    'management.automation.psreference',
    'ref',
})


#: The out-variable common parameters PowerShell defines on every advanced command. Naming them here
#: floors the derivation below the way `_canonical_type_name` floors the type allow-list: each is a
#: fixed engine contract whose loss silences a real write, so a collected surface that no longer
#: carries one fails the load rather than letting `_WRITING_PARAMETERS` shrink to `{setseed}` and
#: quietly stop treating `-OutVariable` as an out-parameter.
_REQUIRED_OUT_VARIABLE_PARAMETERS = frozenset({
    'errorvariable',
    'informationvariable',
    'outvariable',
    'pipelinevariable',
    'warningvariable',
})


def _writing_parameters() -> frozenset[str]:
    """
    The parameter spellings whose presence makes a command write, however pure the transform it
    names, gathered once at import. The out-variable common parameters — `-OutVariable`,
    `-ErrorVariable` and the rest — bind their argument as the *name* of a variable the command
    fills, so `Get-Date -OutVariable d` sets `$d` and is an out-parameter in cmdlet clothing, no
    more removable than `[Int]::TryParse($s, [ref]$n)`. They are read from the collected
    common-parameter surface rather than hardcoded, with their aliases (`ov`, `ev`, ...), so the
    set tracks what PowerShell reports: a common parameter names a variable exactly when its name
    ends in `Variable`, the convention the engine defines them under, which `-OutBuffer` (a count)
    is the one common parameter to fail. `-SetSeed` is added on top — not a common parameter but a
    `Get-Random` switch that rewrites the session's generator state, the same kind of hidden write.
    The match in `_is_writing_parameter` accepts unambiguous abbreviations of every spelling here.
    The derivation is floored by `_REQUIRED_OUT_VARIABLE_PARAMETERS`, so a collected surface that
    drops one of those parameters fails the load rather than shrinking the set silently.
    """
    names = {'setseed'}
    for common, aliases in data.COMMON_PARAMETERS.items():
        if common.endswith('variable'):
            names.add(common)
            names.update(aliases)
    missing = _REQUIRED_OUT_VARIABLE_PARAMETERS - names
    if missing:
        raise ValueError(
            F'the collected common parameters no longer surface the out-variable parameters '
            F'{sorted(missing)!r}; the write-parameter set would silently stop treating them as '
            F'writes, so the data and this module are out of step.'
        )
    return frozenset(names)


_WRITING_PARAMETERS = _writing_parameters()

#: The binary operators whose evaluation writes the automatic `$Matches` variable, so an expression
#: built on one is a store to engine state rather than a value. Every case-sensitivity and negation
#: spelling is listed, because the engine populates `$Matches` for all of them.
_MATCH_OPERATORS = frozenset({
    '-cmatch',
    '-cnotmatch',
    '-imatch',
    '-inotmatch',
    '-match',
    '-notmatch',
})

#: The expression forms that are literally their own value: the base case of `is_side_effect_free`.
_LITERAL_EXPRESSIONS = (
    Ps1HereString,
    Ps1IntegerLiteral,
    Ps1RealLiteral,
    Ps1StringLiteral,
)

#: The expression forms that provably hold no code, read by `_cannot_be_a_scriptblock`. This asks a
#: different question from `_LITERAL_EXPRESSIONS` and is a table of its own even though the two
#: coincide today, because their correct extensions differ: an expandable string can never be a
#: scriptblock and belongs here, while `is_side_effect_free` may not grant one wholesale, since
#: `"$(Start-Process x)"` runs a command. Sharing one table would turn either extension into a
#: silent grant on the other question.
_NON_BLOCK_EXPRESSIONS = (
    Ps1HereString,
    Ps1IntegerLiteral,
    Ps1RealLiteral,
    Ps1StringLiteral,
)

#: The expression forms that can never be read as the name of a member, used by `_invokes_a_member`.
#: The polarity is deliberately the opposite of the two tables above: a form that is *absent* here
#: is treated as a member name, so extending an allow-list elsewhere can never quietly turn a member
#: invocation into a proof of purity. Only numbers qualify — every string form spells a member name
#: however it is quoted, and a here-string named one that `Ps1StringLiteral` alone did not catch.
_NON_MEMBER_EXPRESSIONS = (
    Ps1IntegerLiteral,
    Ps1RealLiteral,
)


def _is_writing_parameter(name: str) -> bool:
    """
    Whether a command parameter, as written in the source, names one of `_WRITING_PARAMETERS`.

    The leading dash is part of the parsed name and is stripped here. PowerShell also binds any
    unambiguous abbreviation of a parameter, so `-OutVar` is `-OutVariable` and has to be recognized
    as one: the match is a prefix test, not equality. An abbreviation short enough to be ambiguous
    is a runtime error in PowerShell, so rejecting it here costs nothing.
    """
    name = name.lstrip('-').lower()
    return bool(name) and any(parameter.startswith(name) for parameter in _WRITING_PARAMETERS)


def _denotes_shared_storage(node) -> bool:
    """
    Whether an expression denotes storage that something outside it can already reach: a variable, a
    property, or an array slot. A literal, a constructed array and a call result are temporaries —
    the expression that produced them is the only holder — so mutating one of those is unobservable
    while mutating shared storage is a side effect.

    This is what separates `[Array]::Reverse('ab'.ToCharArray())`, a junk statement whose result
    nothing can read, from `[Array]::Reverse($buffer)`, which rewrites a live variable.
    """
    while True:
        if isinstance(node, Ps1ParenExpression):
            node = node.expression
        elif isinstance(node, Ps1CastExpression):
            node = node.operand
        else:
            break
    return isinstance(node, (Ps1Variable, Ps1MemberAccess, Ps1IndexExpression))


def _is_writable_reference(node) -> bool:
    """
    Whether an argument hands the callee a `[ref]` to storage it can write back through. A method
    taking one is an out-parameter API — `[Int]::TryParse($s, [ref]$n)` assigns `$n` — so it mutates
    the caller's state no matter how pure the transformation itself is. Every `TryParse` on the
    numeric, date and network types takes one, which is why this is a rule about the argument rather
    than an entry per method.

    Only the syntactic form is recognized. A reference stashed in a variable first
    (`$r = [ref]$n` and then `[Int]::TryParse($s, $r)`) needs dataflow to see, and treating every
    variable argument as a possible reference would make `[Math]::Max($a, $b)` impure.

    Parentheses are transparent: `([ref]$n)` is how the idiom is most often written, and reading the
    cast only at the top level made the whole call look pure.
    """
    while isinstance(node, Ps1ParenExpression):
        node = node.expression
    return (
        isinstance(node, Ps1CastExpression)
        and normalize_dotnet_type_name(node.type_name) in _REFERENCE_TYPE_NAMES
        and _denotes_shared_storage(node.operand)
    )


def _writes_through_out_parameter(
    type_name: str,
    member: str,
    arguments: Sequence[Expression],
) -> bool:
    """
    Whether a `[Type]::Member(args)` call hands one of its arguments to a by-reference parameter it
    can write back through, which makes it an out-parameter API no purer than `[Int]::TryParse($s,
    $n)` however pure the transform itself is. The collected signature is consulted over the static
    overloads of that arity: if one marks position *i* as `byref` and `args[i]` denotes shared
    storage, the call may assign it.

    This reads the parameter direction from the data, so it catches a bare `$n` handed to an
    out-parameter that `_is_writable_reference` — which sees only the syntactic `[ref]$n` cast —
    misses. Arity is matched exactly: an optional trailing parameter the call omits simply removes
    that overload from consideration, so the match never over-rejects a shorter call.
    """
    for overload in data.static_overloads(type_name, member):
        parameters = overload.get('parameters') or ()
        if len(parameters) != len(arguments):
            continue
        if any(
            parameter['byref'] and _denotes_shared_storage(argument)
            for parameter, argument in zip(parameters, arguments)
        ):
            return True
    return False


_PURE_INSTANCE_METHODS = frozenset({
    'adddays',
    'addhours',
    'addminutes',
    'addmonths',
    'addseconds',
    'addyears',
    'compareto',
    'contains',
    'endswith',
    'equals',
    'gethashcode',
    'gettype',
    'indexof',
    'lastindexof',
    'length',
    'padleft',
    'padright',
    'split',
    'startswith',
    'substring',
    'tochar',
    'tochararray',
    'tolower',
    'tostring',
    'touniversaltime',
    'toupper',
    'trim',
    'trimend',
    'trimstart',
})

_PURE_CMDLETS = _canonical_command_set({
    'get-childitem',
    'get-command',
    'get-content',
    'get-date',
    'get-item',
    'get-location',
    'get-process',
    'get-random',
    'get-variable',
    'measure-object',
    'out-null',
    'out-string',
    'select-object',
    'sort-object',
    'where-object',
})

_PURE_PIPELINE_CMDLETS = _canonical_command_set({
    'foreach-object',
    'select-object',
    'sort-object',
    'where-object',
})

#: Commands that put nothing on their caller's output. Every name here is a command whose whole job
#: is to consume, format, or write elsewhere, so a statement calling one cannot carry the return
#: value of the body it sits in. Three of them emit under `-PassThru`, and which three is read from
#: the shipped parameter data rather than split by hand — `Out-GridView` has the switch while
#: `Export-Csv` and `Export-Clixml` do not, which is the opposite of what the names suggest.
#:
#: `Out-String`, `Write-Output` and `Tee-Object` are deliberately absent, because they emit. Listing
#: one costs recall and never a payload — see `_command_emits_nothing` — but it makes this table
#: claim something false about PowerShell, and the next reader has no way to tell which entries were
#: checked. The direction that does cost a payload is a name *missing* from here.
_SILENT_COMMAND_NAMES = {
    'add-content',
    'export-clixml',
    'export-csv',
    'out-default',
    'out-file',
    'out-gridview',
    'out-host',
    'out-null',
    'out-printer',
    'set-content',
    'write-debug',
    'write-error',
    'write-host',
    'write-information',
    'write-progress',
    'write-verbose',
    'write-warning',
}


def _split_silent_commands() -> tuple[frozenset[str], frozenset[str]]:
    """
    Partition the silent-command table into the names that never emit and the names that emit only
    when `-PassThru` is supplied, reading the switch off the collected parameter data.

    A name whose record declares an output type is rejected: a command cannot both be documented as
    producing values and be listed here as producing none, so `Out-String` declares `System.String`
    and fails the load. The floor reaches only that far, and it is worth being exact about how far
    that is — `Write-Output` and `Tee-Object` declare no output type either, so both load cleanly
    and neither is caught here. What stops them is the behavioral tests, not this check.

    The parameter lookup is case-folded because the collected records are not consistent about it:
    148 of them spell the switch `Passthru`, and an exact test would file the next such name under
    the names that never emit while its `-PassThru` form goes on emitting.
    """
    always: set[str] = set()
    gated: set[str] = set()
    for name in _canonical_command_set(_SILENT_COMMAND_NAMES):
        if data.command_output_types(name):
            raise ValueError(
                F'the PowerShell silent-command table names {name!r}, whose collected record '
                F'declares output types; a command that emits cannot be listed as one that does '
                F'not, so the data and the table are out of step.'
            )
        parameters = {p.lower() for p in data.command(name)['parameters']}
        target = gated if 'passthru' in parameters else always
        target.add(name)
    return frozenset(always), frozenset(gated)


_SILENT_COMMANDS, _SILENT_COMMANDS_UNLESS_PASSTHRU = _split_silent_commands()


def _argument_values(cmd: Ps1CommandInvocation) -> Iterator[Expression | None]:
    """
    The expression behind every argument of a command, named or positional. A switch parameter
    carries no value and yields `None`.
    """
    for arg in cmd.arguments:
        yield arg.value if isinstance(arg, Ps1CommandArgument) else arg


def _scriptblock_arguments(cmd: Ps1CommandInvocation) -> list[Ps1ScriptBlock]:
    """
    The literal scriptblocks a command is handed, named or positional.
    """
    return [value for value in _argument_values(cmd) if isinstance(value, Ps1ScriptBlock)]


def _arguments_are_pure(
    arguments: Sequence[Expression],
    oracle: TypeOracle,
) -> bool:
    """
    Whether an argument list is safe to evaluate *and* hands the callee nothing to write back
    through. Both halves have to hold for every call, so they are asked in one place: a `[ref]`
    argument is side-effect free to evaluate and still makes the call an out-parameter API.
    """
    return all(
        not _is_writable_reference(a) and is_side_effect_free(a, oracle)
        for a in arguments
    )


def _command_arguments_are_pure(
    cmd: Ps1CommandInvocation,
    oracle: TypeOracle,
) -> bool:
    """
    Whether every non-scriptblock argument of a command is side-effect free. A cmdlet being a pure
    transform says nothing about what its operands cost to evaluate, so
    `Out-String -InputObject (Start-Process x)` is as impure as the call it is handed. Scriptblock
    arguments are excluded because binding one does not run it; that is `_command_body_is_pure`.

    A parameter that `_is_writing_parameter` names is rejected whatever its argument evaluates to:
    the argument is a variable *name* the command writes, not a value it reads. A splatted argument
    is rejected for the same reason one step removed — `Get-Date @options` supplies parameters that
    are not in the source at all, so it can carry `-OutVariable` as easily as `-Format` and there is
    nothing here to judge.
    """
    for arg in cmd.arguments:
        if isinstance(arg, Ps1CommandArgument) and _is_writing_parameter(arg.name):
            return False
        value = arg.value if isinstance(arg, Ps1CommandArgument) else arg
        if value is None or isinstance(value, Ps1ScriptBlock):
            continue
        if isinstance(value, Ps1Variable) and value.splatted:
            return False
        if _is_writable_reference(value) or not is_side_effect_free(value, oracle):
            return False
    return True


def _cannot_be_a_scriptblock(value) -> bool:
    """
    Whether an argument provably does not carry a scriptblock the command could run. Only literals
    qualify: a variable, a member access or a call result is whatever it was assigned at runtime,
    and a pipeline cmdlet hands exactly such an argument to the engine to invoke per input item.
    """
    return isinstance(value, _NON_BLOCK_EXPRESSIONS)


def _may_name_a_member(value) -> bool:
    """
    Whether an argument could be the string that names the member a `ForEach-Object` invokes. Read
    through `_NON_MEMBER_EXPRESSIONS`, so an unrecognized form counts as a member name rather than
    as proof there is none.
    """
    if value is None or isinstance(value, Ps1ScriptBlock):
        return False
    return not isinstance(value, _NON_MEMBER_EXPRESSIONS)


def _block_runs_only_its_body(block: Ps1ScriptBlock) -> bool:
    """
    Whether every statement a scriptblock runs is one that `refinery.lib.scripts.ps1.ast.get_body`
    reports. A `begin`/`process`/`end` block and a `param` block are code that it does not report —
    the parser fills either those or `body`, never both — so a caller that judges a block by `body`
    alone judges an empty list and proves nothing about `| ForEach-Object { end { Remove-Item $p }}`
    or `| ForEach-Object { param($p = (Start-Process x)) [Void]$_ }`. This is the hole that
    `body_is_inert` guards for a function body, asked of a block handed to a command.
    """
    return not get_named_blocks(block) and get_param_block(block) is None


def _invokes_a_member(cmd: Ps1CommandInvocation) -> bool:
    """
    Whether a `ForEach-Object` carries its work as a member to call rather than as a scriptblock.
    The member is named by a string argument — `-MemberName Kill` and the positional
    `| ForEach-Object Kill` are the same call — and nothing static says what that member does, so no
    inspection of the blocks beside it proves anything about it.

    The question is therefore asked of the arguments, never of whether a block happened to be seen:
    `| ForEach-Object { [Void]$_ } -MemberName Delete` has a block *and* invokes a member, and
    reading the answer off the block is what let a discarding body vouch for the deletion sitting
    next to it.

    A `ForEach-Object` with no scriptblock at all is the same answer with the argument unread: there
    is no body to prove anything from.

    The parser reports `-Name value` as a switch followed by a positional argument and binds no
    values to parameter names, so the argument a member name sits in is not knowable here. Every
    non-numeric argument therefore counts, and `| ForEach-Object { [Void]$_ } -ErrorAction Stop`
    is rejected along with the member forms. That over-rejection keeps junk; distinguishing the two
    needs the parameter positions and types that `refinery.lib.scripts.ps1.data` does not carry.
    """
    name = get_command_name(cmd)
    if name is None or name.lower() != 'foreach-object':
        return False
    if not any(isinstance(value, Ps1ScriptBlock) for value in _argument_values(cmd)):
        return True
    return any(_may_name_a_member(value) for value in _argument_values(cmd))


def _runs_only_visible_blocks(cmd: Ps1CommandInvocation) -> bool:
    """
    Whether every piece of work a pipeline cmdlet could run is a literal scriptblock this module can
    read. These cmdlets take their work through their arguments, so an argument that is neither a
    readable block nor provably blockless hides code: `| Where-Object $filter` and
    `| ForEach-Object { [Void]$_ } -End $sb` both run whatever the variable holds,
    `_invokes_a_member` covers the member form, and a block whose statements sit in a named or
    `param` block is one `_block_runs_only_its_body` refuses to call readable.

    Every caller that judges such a command by the blocks it can see has to ask this first, or it
    decides on a body it was never shown.
    """
    if _invokes_a_member(cmd):
        return False
    return all(
        value is None
        or _cannot_be_a_scriptblock(value)
        or (isinstance(value, Ps1ScriptBlock) and _block_runs_only_its_body(value))
        for value in _argument_values(cmd)
    )


def _command_body_is_pure(
    cmd: Ps1CommandInvocation,
    oracle: TypeOracle,
) -> bool:
    """
    Check whether all script block arguments of a pipeline cmdlet (ForEach-Object, Where-Object,
    etc.) have side-effect-free bodies. These cmdlets are pure transforms: they evaluate a script
    block per input item without mutating state themselves.

    A scriptblock body is a sequence of statements, so it is `statement_effect` that decides, not
    the expression-level `is_side_effect_free`: a body of `$Null = <pure>` or `[Void]<pure>`
    discards is as harmless as one of bare pure expressions, and only the statement layer knows
    that. The mutual recursion between the two terminates because a body is strictly nested inside
    the command it belongs to.

    The blocks have to be all of the work to be worth reading, which is `_runs_only_visible_blocks`.
    A command that also hides work behind an argument proves nothing here however pure its visible
    bodies are.
    """
    if not _runs_only_visible_blocks(cmd):
        return False
    return not any(
        statement_effect(stmt, oracle) is StatementEffect.EFFECT
        for block in _scriptblock_arguments(cmd)
        for stmt in block.body
    )


def _reflection_read_is_pure(type_key: str, member: str) -> bool:
    """
    Whether reading `member` off a value of the single .NET type `type_key` runs no code and cannot
    throw. An uncollected type is never pure: nothing is known about its surface, so a getter that
    shells out cannot be ruled out. On a collected type an Extended Type System member runs arbitrary
    PowerShell and is impure. An *instance* field is a bare memory slot with no getter, so reading
    one is pure whatever the type. A *static* field is not: its first read runs the declaring type's
    static constructor (`beforefieldinit` relaxes the timing, not the fact), so it is gated like a
    property — as is a property, whose getter may run code (`Process.Path` shells out). Both are pure
    only when the whole read surface is granted (`_PURE_READ_TYPES`) or the specific read is
    (`_PURE_READS`). A member the type does not carry reads as `$null` and is pure — but only on a
    sealed value type, because a candidate that is an imprecise supertype would otherwise vouch for a
    read its runtime subtype does carry.
    """
    record = data.member_record(type_key, member)
    if record is data.MemberLookup.UNCOLLECTED:
        return False
    if record is data.MemberLookup.ABSENT:
        return type_key in _PURE_READ_TYPES
    if record['source'] == 'ets':
        return False
    if record['kind'] == 'field' and record.get('static') is False:
        return True
    if record['kind'] in ('field', 'property'):
        return type_key in _PURE_READ_TYPES or (type_key, member.lower()) in _PURE_READS
    return False


def _member_read_is_pure(obj, member: str, oracle: TypeOracle) -> bool:
    """
    Whether reading the named `member` off `obj` has no side effect, decided over every type the
    `oracle` says `obj` could carry. The read is pure only when it is pure for *all* of them: a
    method return or cmdlet output may be one of several types, and a value that could be any of them
    must be safe whichever it is. An empty candidate set — `obj`'s type is unknown — is never pure,
    since an unknown object could be a type whose getter runs code.
    """
    candidates = oracle.candidate_types(obj)
    return bool(candidates) and all(
        _reflection_read_is_pure(candidate, member) for candidate in candidates
    )


def _grant(verdict: bool, node, oracle: TypeOracle) -> bool:
    """
    A present-member or present-type purity grant, conditioned on the type world being closed at
    `node`. Every branch of `is_side_effect_free` that returns `True` because a member, static
    method or constructor resolves to a known-pure .NET operation routes its verdict through here:
    the grant holds only when no code the script runs could have shadowed that member through the
    Extended Type System or remapped its type name through an accelerator, which is
    `refinery.lib.scripts.ps1.analysis.types.TypeOracle.world_closed_at`. An oracle built without a
    world answers `False`, so such a caller keeps the access. An impurity
    *deny* is never routed through here; it holds unconditionally.
    """
    return verdict and oracle.world_closed_at(node)


def is_side_effect_free(node, oracle: TypeOracle) -> bool:
    """
    Conservative check: return `True` only when evaluating `node` is guaranteed to produce no
    observable side effects beyond yielding a value. The `oracle` types the object of a member read
    so the member gate can decide whether the read runs code; without one it resolves only the
    static surface, and every member read whose object it cannot type stays impure.
    """
    if isinstance(node, _LITERAL_EXPRESSIONS):
        return True
    if isinstance(node, Ps1TypeExpression):
        return True
    if isinstance(node, Ps1Variable):
        return True
    if isinstance(node, Ps1ParenExpression):
        return node.expression is None or is_side_effect_free(node.expression, oracle)
    if isinstance(node, Ps1CastExpression):
        # A cast is a conversion the engine performs by calling into the target type, so it is a
        # present-type grant like any other: a remapped accelerator invalidates it, and a name the
        # metadata cannot resolve is not a type this analysis knows anything about. `Add-Type`, a
        # PowerShell `class` and `[Reflection.Assembly]::Load` all make such a name denote code —
        # PowerShell converts a string to it by running a constructor — so granting on the operand
        # alone deleted the call. Resolving is necessary here, not sufficient: a conversion to a
        # collected type can still run code (`[xml]$s` parses, and follows external DTDs), which
        # `_PURE_CAST_TYPES` is the eventual answer to. The world is read before either check
        # because it is a stored bool that can only veto, while both checks below walk.
        if not oracle.world_closed_at(node):
            return False
        if data.resolve_type(node.type_name) is None:
            return False
        return is_side_effect_free(node.operand, oracle)
    if isinstance(node, Ps1UnaryExpression):
        if node.operator in ('++', '--'):
            return False
        return is_side_effect_free(node.operand, oracle)
    if isinstance(node, Ps1BinaryExpression):
        # The regex operators write the automatic `$Matches`, which the statements after them read;
        # that is a store to shared engine state, not a value the expression merely yields, so the
        # operator has to be read and not just the operands. Deleting `$s -match 'p(.*)q'` left the
        # `$Matches[1]` that carries the payload reading an unset variable.
        if node.operator.lower() in _MATCH_OPERATORS:
            return False
        return is_side_effect_free(node.left, oracle) and is_side_effect_free(node.right, oracle)
    if isinstance(node, Ps1RangeExpression):
        return is_side_effect_free(node.start, oracle) and is_side_effect_free(node.end, oracle)
    if isinstance(node, Ps1ArrayLiteral):
        return all(is_side_effect_free(e, oracle) for e in node.elements)
    if isinstance(node, Ps1HashLiteral):
        return all(
            is_side_effect_free(key, oracle) and is_side_effect_free(value, oracle)
            for key, value in node.pairs
        )
    if isinstance(node, Ps1ArrayExpression):
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, Ps1ExpressionStatement) and stmt.expression is not None:
                return is_side_effect_free(stmt.expression, oracle)
        return len(node.body) == 0
    if isinstance(node, Ps1IndexExpression):
        # Indexing selects the `Item` member, so it is the bracket spelling of the member read
        # below and carries the same Extended Type System exposure.
        pure = is_side_effect_free(node.object, oracle) and is_side_effect_free(node.index, oracle)
        return _grant(pure, node, oracle)
    if isinstance(node, Ps1MemberAccess):
        # A read is side-effect free only when the object is pure to evaluate *and* selecting the
        # member runs no code. Returning the object's own purity was the fail-open shape this gate
        # replaces: it deleted `(Get-Process).Path`, an Extended Type System getter that shells out,
        # because the pipeline that produced the object was itself pure. A literal member name —
        # bare (`.Path`) or quoted (`.'Path'`) — names one member the gate can check; a computed
        # member name (`$x.$(...)`) leaves the selected member unknown, so a read through it can
        # never be proven pure however pure the name expression is.
        if not is_side_effect_free(node.object, oracle):
            return False
        member = get_member_name(node.member)
        if member is None:
            return False
        return _grant(_member_read_is_pure(node.object, member, oracle), node, oracle)
    if isinstance(node, Ps1InvokeMember):
        if not _arguments_are_pure(node.arguments, oracle):
            return False
        if node.access == Ps1AccessKind.STATIC:
            obj = node.object
            member = node.member
            # A computed or quoted member name (`[IO.Path]::$m()`, `[IO.Path]::'GetTempFileName'()`)
            # cannot be matched against the carve-outs, so the whole-type grant below must not fire
            # for it either — that is how an obfuscated call reaches the one writing member of an
            # otherwise pure type.
            if isinstance(obj, Ps1TypeExpression) and isinstance(member, str):
                # The type name is resolved through the collected metadata, not truncated, so every
                # spelling of a type lands on one canonical key, and a type the data does not
                # describe resolves to nothing and falls through to impure: the fail-closed default.
                resolved = data.resolve_type(obj.name)
                if resolved is not None:
                    type_key = resolved.lower()
                    key = (type_key, member.lower())
                    if key in _IMPURE_STATIC_METHODS:
                        return False
                    if key in _MUTATING_STATIC_METHODS:
                        pure = not any(_denotes_shared_storage(a) for a in node.arguments)
                        return _grant(pure, node, oracle)
                    if _writes_through_out_parameter(obj.name, member, node.arguments):
                        return False
                    if type_key in _PURE_STATIC_METHOD_TYPES:
                        return _grant(True, node, oracle)
                    if key in _PURE_STATIC_METHODS:
                        return _grant(True, node, oracle)
        elif is_side_effect_free(node.object, oracle):
            member = node.member
            if isinstance(member, str) and member.lower() in _PURE_INSTANCE_METHODS:
                return _grant(True, node, oracle)
        return False
    if isinstance(node, Ps1CommandInvocation):
        if node.redirections:
            return False
        new_object = extract_new_object(node)
        if new_object is not None:
            if not oracle.may_trust_command_name('new-object', node):
                return False
            type_name, ctor_args = new_object
            resolved = data.resolve_type(type_name)
            if resolved is not None and resolved.lower() in _PURE_STATIC_METHOD_TYPES:
                return _grant(_arguments_are_pure(ctor_args, oracle), node, oracle)
            return False
        name = get_command_name(node)
        if name is None:
            return False
        name = name.lower()
        # A command the script redefines no longer runs what the metadata describes, and neither
        # does any command in a script able to rebind names, so its purity is not the built-in's.
        if not oracle.may_trust_command_name(name, node):
            return False
        # The pipeline set is checked through the same gate rather than after the plain one: three
        # of its four members are in both, so testing the plain set first would make the body check
        # below unreachable for `Where-Object`, `Select-Object` and `Sort-Object`.
        if name not in _PURE_CMDLETS and name not in _PURE_PIPELINE_CMDLETS:
            return False
        if not _command_arguments_are_pure(node, oracle):
            return False
        # Routed through `_grant` like every other grant, though `may_trust_command_name` above
        # already refuses an open world. The redundancy is the point: this arm would otherwise hold
        # its world check inside a name-trust question, so narrowing that question back to what its
        # name suggests would silently reopen a fail-open hole with nothing in the path to catch it.
        if name in _PURE_PIPELINE_CMDLETS:
            return _grant(_command_body_is_pure(node, oracle), node, oracle)
        return _grant(True, node, oracle)
    if isinstance(node, Ps1Pipeline):
        return all(
            isinstance(el, Ps1PipelineElement)
            and not el.redirections
            and is_side_effect_free(el.expression, oracle)
            for el in node.elements
        )
    if isinstance(node, Ps1ExpandableString):
        return all(is_side_effect_free(p, oracle) for p in node.parts)
    return False


def is_pure_constant(node) -> bool:
    """
    Whether an expression is a side-effect-free constant that can be removed as a standalone
    statement: a numeric literal or one of the built-in constants `$Null`, `$True`, `$False`,
    through any enclosing parentheses and unary sign. String literals are excluded because they may
    be intentional pipeline output.

    This is a strict refinement of `StatementEffect.OUTPUT`: an expression statement whose
    expression is a pure constant always classifies as `OUTPUT`. The two pruning passes therefore
    have nested candidate sets rather than independently drifting ones — the dead-code pass, which
    prunes only constants, is provably the more conservative of the two.
    """
    if isinstance(node, (Ps1IntegerLiteral, Ps1RealLiteral)):
        return True
    if is_builtin_variable(node):
        return True
    if isinstance(node, Ps1ParenExpression):
        return is_pure_constant(node.expression)
    if isinstance(node, Ps1UnaryExpression) and node.operator in ('+', '-'):
        return is_pure_constant(node.operand)
    return False


class StatementEffect(enum.Enum):
    """
    The observable effect of evaluating a standalone statement, used by every pass that decides
    whether a statement can be pruned from a body:

    - `EFFECT`: the statement performs a side effect (a command call, a store to a real variable, an
      increment); it must be preserved.
    - `OUTPUT`: the statement is side-effect-free but yields a value to the enclosing pipeline (a
      bare constant, a pure expression); it is junk at a discarding position, but in a captured body
      it may be the return value, so removing it needs an emit-safety check.
    - `DISCARD`: the statement is a syntactic no-op that yields nothing and does nothing observable
      (an empty statement, the `$Null = <pure>` and `[Void]<pure>` discard idioms, an `Out-Null`
      pipeline, a discarding `ForEach`); it is always safe to remove, even when it empties the body.

    A discard idiom throws away a *value*, never the work that produced it: every one of them is
    recognized only over an operand that `is_side_effect_free` accepts, so `[Void](Start-Process x)`
    is an `EFFECT` like any other call.

    `EFFECT` deliberately says nothing about emission, and splitting it into an emitting and a
    silent member would not pay: a silent `EFFECT` is still un-removable, so every consumer would
    grow a branch to reach the verdict it already reaches. Whether a statement emits is a second and
    orthogonal fact, asked through `output_is_covered` — `Write-Host x` and `Get-Item x` are both
    `EFFECT` and only the second can carry a body's return value.
    """
    EFFECT = 'effect'
    OUTPUT = 'output'
    DISCARD = 'discard'


def _is_void_cast(node) -> TypeGuard[Ps1CastExpression]:
    """
    Whether a node is a cast to `[Void]`, the discard idiom that throws a value away. The type name
    is folded through `refinery.lib.scripts.ps1.ast.normalize_dotnet_type_name` so that the
    `[System.Void]` spelling an obfuscator emits is the same idiom.
    """
    return (
        isinstance(node, Ps1CastExpression)
        and normalize_dotnet_type_name(node.type_name) == 'void'
    )


def _is_null_discard(node) -> TypeGuard[Ps1AssignmentExpression]:
    """
    Whether a node is the `$Null = ...` discard idiom, which evaluates its right-hand side and puts
    nothing on the output.
    """
    return (
        isinstance(node, Ps1AssignmentExpression)
        and node.operator == '='
        and is_builtin_variable(node.target, {'null'})
    )


def statement_effect(stmt, oracle: TypeOracle) -> StatementEffect:
    """
    Classify the observable effect of a standalone statement as a `StatementEffect`. This is the one
    shared authority the dead-code and junk-removal passes consult so they never disagree about
    whether a statement carries a body's output: a `DISCARD` emits nothing and can always be
    dropped, an `OUTPUT` yields a value that emit-safety must protect in a captured body, and an
    `EFFECT` must always be kept.
    """
    if not isinstance(stmt, Ps1ExpressionStatement):
        return StatementEffect.EFFECT
    expr = stmt.expression
    if expr is None:
        return StatementEffect.DISCARD
    if _is_void_cast(expr):
        if is_side_effect_free(expr.operand, oracle):
            return StatementEffect.DISCARD
        return StatementEffect.EFFECT
    if isinstance(expr, Ps1Pipeline):
        # The prefix is walked exactly once and every branch below is derived from that one answer.
        # Asking `_pipeline_prefix_is_pure` per idiom and then falling through to
        # `is_side_effect_free(expr)` re-walks the same elements, and because a pipeline cmdlet
        # body re-enters here through `_command_body_is_pure`, that doubling compounds into 2^depth
        # work on the nested `... | ForEach-Object { ... } | Out-Null` shape.
        prefix_is_pure = _pipeline_prefix_is_pure(expr, oracle)
        if prefix_is_pure and (
            _pipeline_ends_with_out_null(expr, oracle)
            or _pipeline_ends_with_void_foreach(expr, oracle)
        ):
            return StatementEffect.DISCARD
        if _pipeline_ends_with_cmdlet(expr, _PURE_PIPELINE_CMDLETS):
            # A pure pipeline cmdlet (`... | Where-Object {...}`) yields a filtered value a caller
            # may consume, so it is kept even though it performs no side effect of its own.
            return StatementEffect.EFFECT
        if prefix_is_pure and _pipeline_final_is_pure(expr, oracle):
            return StatementEffect.OUTPUT
        return StatementEffect.EFFECT
    if _is_null_discard(expr):
        if expr.value is not None and is_side_effect_free(expr.value, oracle):
            return StatementEffect.DISCARD
        return StatementEffect.EFFECT
    if is_side_effect_free(expr, oracle):
        return StatementEffect.OUTPUT
    return StatementEffect.EFFECT


def _terminal_invocation(pipeline: Ps1Pipeline) -> Ps1CommandInvocation | None:
    """
    The unredirected command invocation that terminates a multi-element pipeline, else `None`. A
    single-element pipeline has no terminator in this sense: there is no upstream value for it to
    consume.
    """
    if len(pipeline.elements) < 2:
        return None
    last = pipeline.elements[-1]
    if not isinstance(last, Ps1PipelineElement) or last.redirections:
        return None
    expr = last.expression
    if not isinstance(expr, Ps1CommandInvocation) or expr.redirections:
        return None
    return expr


def _terminal_command(pipeline: Ps1Pipeline, name: str) -> Ps1CommandInvocation | None:
    """
    The invocation that terminates a pipeline when it is an unredirected call to `name`, written
    under exactly that spelling, else `None`.
    """
    expr = _terminal_invocation(pipeline)
    if expr is None:
        return None
    command = get_command_name(expr)
    if command is None or command.lower() != name:
        return None
    return expr


def _terminal_command_resolved(pipeline: Ps1Pipeline, name: str) -> Ps1CommandInvocation | None:
    """
    The invocation that terminates a pipeline when it *resolves* to `name`, so `%` and `foreach`
    reach `foreach-object` the way the engine does.

    Deliberately a second function rather than a flag on `_terminal_command`, because which of the
    two a caller may use is decided by what its answer licenses. This one feeds
    `_statement_can_emit`, where a match withholds a deletion, so following an alias can only keep
    more code. `_terminal_command`'s callers reach `StatementEffect.DISCARD`, which deletes, and
    they ask `may_trust_command_name` under the canonical name — a script defining
    `function foreach` shadows the spelling `foreach`, which a question about `foreach-object`
    cannot see, so resolving there would grant exactly the deletion the shadow set holds evidence
    against.
    """
    expr = _terminal_invocation(pipeline)
    if expr is None or resolve_command_name(expr) != name:
        return None
    return expr


def _pipeline_sink_discards_its_input(pipeline: Ps1Pipeline) -> bool:
    """
    Whether the pipeline's terminator throws away everything that reaches it, so the statement puts
    nothing on the enclosing body's output.

    This is the shape question alone. What the terminator costs to evaluate is a separate matter and
    belongs to `statement_effect`: `... | Out-Null -InputObject (Start-Process x)` runs a call and
    still emits nothing, so it is an `EFFECT` that cannot carry a body's return value. Conflating
    the two is what let a non-emitting survivor stand in for the value a `RETURNING` body exists to
    produce.
    """
    if _terminal_command_resolved(pipeline, 'out-null') is not None:
        return True
    foreach = _terminal_command_resolved(pipeline, 'foreach-object')
    if foreach is None:
        return False
    blocks = _scriptblock_arguments(foreach)
    return bool(blocks) and all(
        not _statement_can_emit(stmt) for block in blocks for stmt in block.body
    )


def _pipeline_ends_with_out_null(
    pipeline: Ps1Pipeline,
    oracle: TypeOracle,
) -> bool:
    """
    Whether a pipeline is terminated by an `Out-Null` that throws its input away *and* costs nothing
    to reach. The terminator's own arguments are part of the question:
    `... | Out-Null -InputObject (Start-Process x)` runs the call it is handed, so it discards a
    value the pipeline never carried and is not a junk sink.
    """
    out_null = _terminal_command(pipeline, 'out-null')
    if out_null is None or not oracle.may_trust_command_name('out-null', out_null):
        return False
    return _command_arguments_are_pure(out_null, oracle)


def _pipeline_prefix_is_pure(
    pipeline: Ps1Pipeline,
    oracle: TypeOracle,
) -> bool:
    for el in pipeline.elements[:-1]:
        if not isinstance(el, Ps1PipelineElement) or el.redirections:
            return False
        if not is_side_effect_free(el.expression, oracle):
            return False
    return True


def _pipeline_final_is_pure(
    pipeline: Ps1Pipeline,
    oracle: TypeOracle,
) -> bool:
    """
    Whether the last element of a pipeline is side-effect free. Together with
    `_pipeline_prefix_is_pure` this is the purity of the whole pipeline, split so that a caller
    which already knows about the prefix does not walk it a second time.
    """
    if not pipeline.elements:
        return True
    last = pipeline.elements[-1]
    return (
        isinstance(last, Ps1PipelineElement)
        and not last.redirections
        and is_side_effect_free(last.expression, oracle)
    )


def _pipeline_ends_with_void_foreach(
    pipeline: Ps1Pipeline,
    oracle: TypeOracle,
) -> bool:
    """
    Detect junk pipelines like `... | ForEach-Object { [Void]$_ }` or
    `... | ForEach-Object { $Null = $_ }` where the ForEach body explicitly discards all output.
    These are anti-analysis noise injected into malware scripts. Whether a body statement discards
    is `statement_effect`'s answer rather than a second copy of the idiom table, so a discard of a
    value that is not itself side-effect free does not count — the body

        ForEach-Object { [Void](Start-Process x) }

    discards the result of a call that still happens.

    A `ForEach-Object` that carries work no block accounts for is not a match, whether that work is
    a member to invoke (`| ForEach-Object { [Void]$_ } -MemberName Delete`) or a block a variable
    holds (`| ForEach-Object { [Void]$_ } -End $sb`). That is `_runs_only_visible_blocks`, asked
    here for the same reason `_command_body_is_pure` asks it: the discards below prove a property of
    the blocks they saw, and a body that was never shown is not among them.
    """
    foreach = _terminal_command(pipeline, 'foreach-object')
    if foreach is None or not oracle.may_trust_command_name('foreach-object', foreach):
        return False
    if not _command_arguments_are_pure(foreach, oracle):
        return False
    if not _runs_only_visible_blocks(foreach):
        return False
    blocks = _scriptblock_arguments(foreach)
    return bool(blocks) and all(
        statement_effect(stmt, oracle) is StatementEffect.DISCARD
        for block in blocks
        for stmt in block.body
    )


def _pipeline_ends_with_cmdlet(pipeline: Ps1Pipeline, names: frozenset[str]) -> bool:
    expr = _terminal_invocation(pipeline)
    if expr is None:
        return False
    name = get_command_name(expr)
    return name is not None and name.lower() in names


class BodyRole(enum.Enum):
    """
    How a statement body relates to the code around it — the emission question every pruning pass
    has to answer before it removes anything. A `refinery.lib.scripts.Block` or
    `refinery.lib.scripts.ps1.model.Ps1Code` body is one of:

    - `OPAQUE`: the body's value is captured (an assignment right-hand side, `$(...)`, `@(...)`, a
      stored or argument scriptblock, a piped `&{}`); pruning any statement could destroy an
      observable value, so the body is left untouched.
    - `SCRIPT`: the script root. It has no return value — its output goes to the host — but it must
      never be pruned away entirely, which is what `pruning_erases_body` guards.
    - `RETURNING`: a body whose value the caller observes — a function or method body, or a bare
      `&{ ... }` / `.{ ... }` in statement position. Removing the statement that carries the output
      silences the return value, so pruning goes through `output_observed` and `output_is_covered`.
    - `NESTED`: a plain nested block that runs for its side effects (a loop or `if` body in
      statement position); it has no observable value of its own, so statements may be pruned
      freely.
    """
    OPAQUE = 'opaque'
    SCRIPT = 'script'
    RETURNING = 'returning'
    NESTED = 'nested'


def _scriptblock_is_captured(block: Ps1ScriptBlock) -> bool:
    """
    Return `True` when the value of a `refinery.lib.scripts.ps1.model.Ps1ScriptBlock` is captured
    rather than run for its observable output. A bare `&{ ... }` / `.{ ... }` in statement position
    produces output that the pass may prune into; every other scriptblock (a stored closure
    `$x = { ... }`, an argument block, or an invocation whose result is assigned, passed, or piped)
    is treated as captured and left opaque.
    """
    parent = block.parent
    if isinstance(parent, Ps1FunctionDefinition):
        return False
    if not (isinstance(parent, Ps1CommandInvocation) and parent.name is block):
        return True
    invocation_parent = parent.parent
    if isinstance(invocation_parent, Ps1ExpressionStatement):
        return False
    if isinstance(invocation_parent, Ps1PipelineElement):
        pipeline = invocation_parent.parent
        if (
            isinstance(pipeline, Ps1Pipeline)
            and len(pipeline.elements) == 1
            and isinstance(pipeline.parent, Ps1ExpressionStatement)
        ):
            return False
    return True


def body_role(node) -> BodyRole | None:
    """
    Classify the statement body that `node` owns as a `BodyRole`, or return `None` when `node` owns
    no prunable body — which is also how `@( ... )` stays out of every pruning walk, since
    `refinery.lib.scripts.ps1.ast.get_body` deliberately does not recognize it. Ambiguous capture
    always resolves to `OPAQUE`.

    A plain `refinery.lib.scripts.Block` — a loop, `if`, `try`, `catch`, `finally`, or `trap` body —
    carries no role of its own and derives one by walking outward to the nearest body owner. That
    walk reports the *owner's* role only for a function body, so the same block classifies three
    ways depending on where it sits:

        if ($x) { 1 }                    at script level  ->  NESTED
        function f { if ($x) { 1 } }                      ->  RETURNING
        &{ if ($x) { 1 } }                                ->  NESTED

    A nested block's value is observed exactly when its owner's is, so the consistent answer would
    be the owner's role in all three cases, and `NESTED` is the more permissive one at both the
    script and the `&{}` boundary. The passes have shipped with this behavior and all three traces
    are pinned by test; resolving it needs the reachability of the flow layer, so it is deliberately
    left as it stands rather than changed as a side effect of consolidating the authority here.
    """
    if get_body(node) is None:
        return None
    if isinstance(node, Ps1Script):
        return BodyRole.SCRIPT
    if isinstance(node, Ps1SubExpression):
        return BodyRole.OPAQUE
    if isinstance(node, Ps1ScriptBlock):
        if isinstance(node.parent, Ps1FunctionDefinition) and node.parent.body is node:
            return BodyRole.RETURNING
        return BodyRole.OPAQUE if _scriptblock_is_captured(node) else BodyRole.RETURNING
    prev = node
    cursor = node.parent
    while cursor is not None:
        if isinstance(cursor, (Ps1SubExpression, Ps1ArrayExpression, Ps1DataSection)):
            return BodyRole.OPAQUE
        if isinstance(cursor, Ps1AssignmentExpression) and cursor.value is prev:
            return BodyRole.OPAQUE
        if isinstance(cursor, Ps1ScriptBlock):
            if _scriptblock_is_captured(cursor):
                return BodyRole.OPAQUE
            if isinstance(cursor.parent, Ps1FunctionDefinition) and cursor.parent.body is cursor:
                return BodyRole.RETURNING
            return BodyRole.NESTED
        if isinstance(cursor, Ps1Script):
            return BodyRole.NESTED
        prev = cursor
        cursor = cursor.parent
    return BodyRole.NESTED


def output_observed(role: BodyRole) -> bool:
    """
    Whether a body of this role has a return value that pruning must protect. True only for
    `BodyRole.RETURNING`: a `NESTED` body has no observable value, the `SCRIPT` root has no return
    value, and an `OPAQUE` body is never pruned at all.
    """
    return role is BodyRole.RETURNING


def fault_is_observed(stmt: Node) -> bool:
    """
    Whether removing `stmt` could change whether an enclosing handler runs: it sits directly in the
    `try` block of a `try`/`catch` with at least one catch clause that does something.

    `StatementEffect` models emission and side effect, not fault — nothing here can answer whether a
    statement throws. So a statement is removed from a protected body only when it is not protected
    at all, however pure it looks: `[Int]'abc'` produces no output and raises, and dropping it makes
    the `try` body empty, which is evidence about this pass rather than about the code as written.

    This is the first of two refusals and no longer the only one.
    `Ps1DeadCodeElimination._prune_try` now declines to dissolve a construct whose handler has a
    body, whatever emptied the `try` block, because the routes to an empty body are many and each
    new pass adds another. Neither refusal makes the other redundant: this one keeps the body from
    being emptied through the pruning path at all, and that one covers every other route.

    An empty `catch { }` is deliberately not a handler that does something: it swallows the error
    and execution continues either way, so removing a throwing statement changes nothing observable.
    That is the shape obfuscators emit, which is why this costs the cleanup passes almost nothing.

    The converse under-deletion is left alone: `_prune_try` requires *every* catch clause to be
    empty before it dissolves a construct, where a body proven not to throw would let it dissolve
    one with a live handler and delete that handler as unreachable. Both directions are the same
    missing axis, and both are the fault axis's to settle.
    """
    block = stmt.parent
    if block is None:
        return False
    guard = block.parent
    if not isinstance(guard, Ps1TryCatchFinally) or guard.try_block is not block:
        return False
    return any(
        clause.body is not None and clause.body.body
        for clause in guard.catch_clauses
    )


_BODY_BEARING_STATEMENTS = (
    Ps1DoLoop,
    Ps1ForEachLoop,
    Ps1ForLoop,
    Ps1IfStatement,
    Ps1SwitchStatement,
    Ps1TryCatchFinally,
    Ps1WhileLoop,
)


def _redirection_takes_output_away(
    redirection: Ps1FileRedirection | Ps1MergingRedirection,
) -> bool:
    """
    Whether a single redirection moves the output stream somewhere the enclosing body cannot see it.
    A file redirection of `OUTPUT` or of `ALL` (`>`, `>>`, `*>`) writes the values to disk; a merge
    carries them into whichever stream it names, so it takes them away exactly when it reads *from*
    output and writes somewhere that is not output.

    The direction is what decides this, and reading the wrong end inverts the answer on the common
    forms: `2>&1` and `3>&1` merge another stream *into* output and leave emission untouched, while
    `1>&2` is the one that silences it. `*>&1` names `ALL` as its source, which includes output
    merged onto itself, so it is not a removal either.
    """
    if isinstance(redirection, Ps1FileRedirection):
        return redirection.stream in (Ps1RedirectionStream.OUTPUT, Ps1RedirectionStream.ALL)
    return (
        redirection.from_stream in (Ps1RedirectionStream.OUTPUT, Ps1RedirectionStream.ALL)
        and redirection.to_stream is not Ps1RedirectionStream.OUTPUT
    )


def _output_is_redirected_away(cmd: Ps1CommandInvocation) -> bool:
    """
    Whether any redirection written on a call moves its output somewhere the enclosing body cannot
    see it. The parser hangs every redirection off the invocation and never off the pipeline element
    around it, so this is the whole of what a statement can carry; only the call whose output would
    reach the body is asked, because an earlier pipeline element's redirection diverts what that
    element hands downstream, which is a different question. Where a merge and a file redirection
    are written together (`2>&1 > C:\\log`), the file redirection is still a removal and any one of
    them is enough to answer yes.
    """
    return any(_redirection_takes_output_away(r) for r in cmd.redirections)


def _passthru_is_requested(cmd: Ps1CommandInvocation) -> bool:
    """
    Whether a call provably asks for `-PassThru`, the switch that turns a writing command into an
    emitting one. Anything short of proof is a no: `-PassThru:$false` binds false, and
    `-PassThru:$env:x` binds something no analysis here can read, so both answer no and the command
    keeps its silent verdict — which withholds a deletion rather than licensing one.

    The leading dash is part of the parsed name and is stripped, but the rest is matched in full
    rather than by prefix — the opposite of `_is_writing_parameter`. There the abbreviation
    PowerShell would accept has to be caught because a match *refuses* a purity grant; here a match
    asserts emission, so accepting `-Pass` would be the direction that deletes a payload.
    """
    for arg in cmd.arguments:
        if not isinstance(arg, Ps1CommandArgument) or arg.name.lstrip('-').lower() != 'passthru':
            continue
        return arg.value is None or is_truthy(arg.value) is True
    return False


def _command_emits_nothing(cmd: Ps1CommandInvocation) -> bool:
    """
    Whether a command puts nothing on the output of the body its call sits in.

    The name is resolved through `refinery.lib.scripts.ps1.ast.resolve_command_name`, so an alias
    (`sc`, `ogv`, `%`) reaches the same entry the plain name does. That is safe in the direction
    this table is read: a hit says "cannot emit", which withholds a deletion, so resolving toward a
    bare name can only keep more code. A script that redefines `Write-Host` is wrong here for the
    same reason and in the same harmless direction — the definition is not consulted, the call reads
    as silent, and the payload beside it survives.

    **The two error directions are not symmetric.** A name listed here in error costs recall: junk
    that would have been deleted is kept. A name *missing* from here costs a payload, because the
    call then reads as carrying the body's output and whatever stands beside it is deleted as
    redundant. The table is far from complete in that sense — `Start-Sleep`, `Remove-Item`,
    `Stop-Process` and every other command that acts without emitting are absent — so this is a
    floor on what is known to be silent and never a decision that anything else emits.

    It is a hand-written list because nothing shipped separates the two: the metadata records that
    `Out-Null` and `Write-Output` have identical `output_types`, `output_type_declared` and `kind`,
    so the command that swallows its input and the one that forwards it are indistinguishable in the
    data. What must not happen is this predicate being re-keyed to feed a *deletion* decision,
    `_pipeline_ends_with_void_foreach` being the tempting one, where a wrong name would delete code
    instead of keeping it.
    """
    name = resolve_command_name(cmd)
    if name is None:
        return False
    if name in _SILENT_COMMANDS:
        return True
    if name not in _SILENT_COMMANDS_UNLESS_PASSTHRU:
        return False
    return not _passthru_is_requested(cmd)


def _emitting_invocation(expr: Expression) -> Ps1CommandInvocation | None:
    """
    The invocation whose output would reach the enclosing body: the call itself when a statement is
    one, else the last element of its pipeline. Unlike `_terminal_invocation` this accepts a
    single-element pipeline and a redirected call, because the question is what the statement emits
    rather than what a sink consumes.
    """
    if isinstance(expr, Ps1CommandInvocation):
        return expr
    if isinstance(expr, Ps1Pipeline) and expr.elements:
        last = expr.elements[-1]
        if not isinstance(last, Ps1PipelineElement):
            return None
        if isinstance(last.expression, Ps1CommandInvocation):
            return last.expression
    return None


def _body_can_emit(block: Block | None) -> bool:
    return block is not None and any(_statement_can_emit(stmt) for stmt in block.body)


def _expression_can_emit(expr: Expression | None) -> bool:
    """
    Whether the value of an expression standing alone as a statement reaches the enclosing body's
    output. Split out of `_statement_can_emit` because `return $x` asks it of a statement that is
    not a `Ps1ExpressionStatement`, and `(Write-Host x)` asks it of an operand one level down.

    A grouping construct is transparent: `(...)` and `$(...)` emit whatever stands inside them, and
    a silent call reads as silent through either. `@(...)` is not one of them — it builds an array
    even when nothing filled it, so `@(Write-Host x)` puts an empty array on the output and keeps
    the permissive answer. Neither is `($x = 1)`: parentheses around an assignment are the one form
    that does put the bound value on the pipeline, so the paren is not unwrapped past one.
    """
    if expr is None:
        return False
    if isinstance(expr, Ps1ParenExpression):
        inner = expr.expression
        if isinstance(inner, Ps1AssignmentExpression):
            return True
        return _expression_can_emit(inner)
    if isinstance(expr, Ps1SubExpression):
        return any(_statement_can_emit(stmt) for stmt in expr.body)
    if _is_void_cast(expr) or isinstance(expr, Ps1AssignmentExpression):
        return False
    invocation = _emitting_invocation(expr)
    if invocation is not None:
        if _output_is_redirected_away(invocation) or _command_emits_nothing(invocation):
            return False
    if isinstance(expr, Ps1Pipeline):
        return not _pipeline_sink_discards_its_input(expr)
    return True


def _statement_can_emit(stmt: Node) -> bool:
    """
    Whether a statement can put a value on the enclosing body's output at all. This is the emission
    question alone, deliberately divorced from what the statement costs to run:
    `[Void](Start-Process x)` cannot carry a body's return value even though `statement_effect`
    calls it an `EFFECT` for the call it wraps, and neither can `... | Out-Null -InputObject (...)`.

    A declaration emits nothing, and neither does an assignment — `$x = 1` binds a value rather than
    yielding one, whatever sits on its right-hand side. A named `data d { ... }` section is an
    assignment too: it binds its block's value to `$d`. Only the unnamed `data { ... }` puts that
    value on the output, and only when its block holds something that emits.

    A statement that holds other statements emits whatever they do, so one is descended into rather
    than granted emission for its shape: `if ($c) { }` and `foreach ($i in $x) { $Null = $i }` put
    nothing anywhere, and answering `True` for them lets an empty branch stand in for the value a
    body exists to produce. Owning a block is enough to be descended into, so a construct the parser
    learns later cannot fall through to the permissive answer by being absent from a list;
    `_BODY_BEARING_STATEMENTS` names the ones that must be descended into *even when they own no
    block at all*, which is what `switch ($a) { }` parses to.

    A `catch` handler is *not* descended into. It runs only when the `try` body faults, so what it
    emits cannot stand in for the value the body produces on the path that does not throw — and
    since the try/catch pass stopped dissolving a construct whose handler has a body, that handler
    is now a survivor sitting beside the payload it would otherwise shadow. The `try` and `finally`
    blocks both run on the normal path and are descended into as usual.

    A jump or an exit leaves the body without putting anything on it; `return $x` is the one that
    carries a value, and it carries exactly what the expression beside it would.

    This is private on purpose. Its safe direction is `False` — see `output_is_covered`, whose one
    production consumer withholds a deletion when nothing covers the output — and that polarity
    lives in the caller rather than here, so a predicate that reads as "can emit" would be a trap to
    reuse anywhere a `True` licenses an action.
    """
    if isinstance(stmt, (
        Ps1ClassDefinition,
        Ps1EnumDefinition,
        Ps1FunctionDefinition,
        Ps1TrapStatement,
    )):
        return False
    if isinstance(stmt, Ps1Jump):
        return False
    if isinstance(stmt, Ps1ReturnStatement):
        return _expression_can_emit(stmt.pipeline)
    if isinstance(stmt, Ps1Exit):
        return False
    if isinstance(stmt, Ps1DataSection):
        return not stmt.name and _body_can_emit(stmt.body)
    blocks = list(_nested_blocks(stmt))
    if blocks or isinstance(stmt, _BODY_BEARING_STATEMENTS):
        return any(_body_can_emit(block) for block in blocks)
    if not isinstance(stmt, Ps1ExpressionStatement):
        return True
    return _expression_can_emit(stmt.expression)


def _nested_blocks(stmt: Node) -> Iterator[Block]:
    """
    The statement blocks that run on the path a body's output travels. A `catch` clause is skipped
    rather than unwrapped: `Ps1CatchClause` is a wrapper carrying a type filter, and the handler
    below it runs only on a fault.
    """
    for child in stmt.children():
        if isinstance(child, Block):
            yield child


def output_is_covered(survivors: Sequence[Node]) -> bool:
    """
    Whether some statement in `survivors` still carries the body's output, so that removing the
    pure-output statements around it cannot silence a `BodyRole.RETURNING` body's return value.

    `survivors` is the surviving statement set itself and never a node to walk up from. A caller may
    hold freshly synthesized statements that are not parented into a body yet, and statements
    hoisted out of a pruned block still point at the block they came from; answering this question
    by walking `parent` is what used to delete live return values.

    The check is coarse: every survivor that can emit at all counts as covering, including a
    conditional that may not execute. It therefore over-counts, permitting a prune that a precise
    analysis would refuse. What it may not do is count a statement that provably emits nothing —
    a definition, an assignment, a discard idiom — because such a survivor would silence the body
    while appearing to cover it. Tightening the rest needs reachability.
    """
    return any(_statement_can_emit(stmt) for stmt in survivors)


def pruning_erases_body(role: BodyRole, survivors: Sequence[Node]) -> bool:
    """
    Whether pruning a body of this role down to `survivors` would erase it: nothing would survive,
    and a body of this role must not become empty. Only the `BodyRole.SCRIPT` root qualifies — a
    script that is nothing but function definitions is a module whose functions may be dot-sourced,
    and a script that is nothing but `42` still emits `42` — so emptying it would delete real code.
    Every other role may legitimately prune to nothing; that is what turns an injected junk function
    inert.

    Like `output_is_covered`, this takes the surviving statement set itself and never walks up from
    a node.
    """
    return not survivors and role is BodyRole.SCRIPT


def _param_block_is_inert(
    block: Ps1ParamBlock | None,
    oracle: TypeOracle,
) -> bool:
    """
    Whether a `param( ... )` block runs nothing when the function is called. Declaring a name binds
    storage and evaluates nothing, but a default value is an expression the engine runs on every
    call that omits the argument, and an attribute is work of its own — a `[ValidateScript({...})]`
    body runs on every call that supplies one, and a `[Parameter(Mandatory)]` makes the call prompt.

    Attributes are rejected wholesale rather than matched against a table: which of them do
    something observable is not a question this module can answer, and a type constraint is the one
    form that provably does not, so it is the only one let through.
    """
    if block is None:
        return True
    if block.attributes:
        return False
    return all(
        not any(isinstance(a, Ps1Attribute) for a in parameter.attributes)
        and (parameter.default_value is None or is_side_effect_free(parameter.default_value, oracle))
        for parameter in block.parameters
    )


def body_is_inert(node, oracle: TypeOracle) -> bool:
    """
    Whether the body that `node` owns neither emits a value nor performs a side effect: `node` is
    `None`, the body is empty, or every statement in it is a `StatementEffect.DISCARD`. An inert
    function body makes the function itself unobservable, so its definition and its bare call sites
    can be dropped together.

    A node that owns a `begin`/`process`/`end` block is never inert: `get_body` reports an empty
    statement list for it, and reading that as "nothing happens here" would delete an advanced
    function together with every call to it. A `param` block is the same hole — `get_body` does not
    report it either, and `function f { param($x = (Start-Process n)) }` runs a command on every
    call that omits the argument — but unlike a named block it is not code by its mere presence, so
    it is judged by `_param_block_is_inert` rather than counted. Anything else `get_body` does not
    recognize is not a body owner and cannot be shown to be inert either.
    """
    if node is None:
        return True
    body = get_body(node)
    if body is None or get_named_blocks(node):
        return False
    if not _param_block_is_inert(get_param_block(node), oracle):
        return False
    return all(statement_effect(stmt, oracle) is StatementEffect.DISCARD for stmt in body)
