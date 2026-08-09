"""
The .NET type and PowerShell command database that the PowerShell analysis and deobfuscation
subsystems share: type accelerators and members, command names, aliases and parameters, WMI classes,
and the small lookup tables built from them. The data is collected by run-pwsh.ps1 on genuine
Windows PowerShell 5.1 and shipped as the compressed `pwsh-*.json.gz` resources.

Two surfaces live here. The `*` tables (`TYPE_MEMBERS`, `KNOWN_CMDLETS`, ...) are the historical
views the deobfuscation transforms read today; they reconstruct the shapes those modules expect. The
functions at the end (`resolve_type`, `canonical_type`, `type_members`, `command`, `member_order`)
are the query API that later work migrates those consumers onto, and they alone reach the richer
facts the new format carries — per-overload signatures, member kind, the true Get-Member order.
"""
from __future__ import annotations

import enum
import gzip
import json
import operator
import re

from refinery.lib.resources import datapath
from refinery.lib.scripts.ps1.dotnet import Ps1TypeName, parse_type_name

SCHEMA_VERSION = 1


def _load(name: str) -> dict:
    with datapath(name).open('rb') as fp:
        return json.loads(gzip.decompress(fp.read()))


_META = _load('pwsh-meta.json.gz')
_schema = _META['schema']['version']
if _schema != SCHEMA_VERSION:
    raise ValueError(
        F'pwsh metadata schema version {_schema} is not the expected {SCHEMA_VERSION}; '
        F'the reader and the collected data are out of step.'
    )

_TYPES = _load('pwsh-types.json.gz')
_COMMANDS = _load('pwsh-commands.json.gz')
_VARIABLES = _load('pwsh-variables.json.gz')
_WMI = _load('pwsh-wmi.json.gz')

#: Type names PowerShell's *parser* understands that are not registered type accelerators, so a
#: collection run on a real host does not report them however authoritative it is — `[ordered]` is
#: recognized in a cast position and denotes an `OrderedDictionary`, but it is absent from the 94
#: accelerators our capture reports because it never was one. Kept beside the collected table rather
#: than injected into it: the data describes a host's .NET surface and this describes the language,
#: and blurring the two would cost the collected table the provenance that makes it trustworthy.
_PARSER_TYPE_KEYWORDS: dict[str, str] = {
    'ordered': 'System.Collections.Specialized.OrderedDictionary',
}

_ACCELERATORS: dict[str, str] = {
    **_PARSER_TYPE_KEYWORDS,
    **{_alias.lower(): _full for _alias, _full in _TYPES['accelerators'].items()},
}
_TYPE_TABLE: dict[str, dict] = _TYPES['types']

#: Commands the capture reports that the host does not have. `Format-Hex` is described here exactly
#: as PowerShell 7 describes it — a `Raw` switch and a `System.String` `Encoding` — and the
#: `Microsoft.PowerShell.Utility` a 5.1 host loads reports version 7.0.0.0, so the collector read it
#: from a shadowing 7.0 module rather than from the host it declares itself authoritative for.
#: Measured three ways on 5.1: `Get-Command Format-Hex` reports CommandNotFoundException, `'ab' |
#: Format-Hex` throws, and the loaded Utility module exports only `Format-Custom`, `Format-List`,
#: `Format-Table` and `Format-Wide`. A probe of thirteen commands only PowerShell 6 or 7 ships found
#: this to be the one that leaked.
#:
#: A record for a command the host cannot run is not inert: the wildcard resolver draws its
#: candidate universe from `KNOWN_CMDLETS`, so `Get-Command Format-H*` had a unique match and was
#: rewritten into a call to a command 5.1 reports as not found — the same defect the `fhx` alias was
#: removed for. Withheld from the derived table rather than deleted from the capture, which stays as
#: collected the way `_PARSER_TYPE_KEYWORDS` above is kept beside it rather than injected into it.
_MISCOLLECTED_COMMANDS = frozenset({'format-hex'})

#: The bare nouns that name a program Windows itself ships, so that 5.1 runs the program rather than
#: retrying the name with a `Get-` prefix — the Application tier, which sits above that retry and
#: which no capture of the session's tables can describe. `tpm` opens `C:\Windows\system32\tpm.msc`,
#: and `Get-Tpm` is what we answered for it.
#:
#: Derived by intersecting every PATHEXT-matching file in `System32` and `Windows` with the 523 bare
#: nouns `refinery.lib.scripts.ps1.ast.implicit_get_retry` rewrites; `tpm` is the whole intersection.
#: A program the analyst installed cannot be here and is the declared residual: `date` resolves to
#: Git's `date.exe` on the development box and to `Get-Date` on one without it, and nothing readable
#: from a script says which. This is a floor under that residual rather than a fix for it.
PROGRAM_NAMES = frozenset({'tpm'})

_COMMAND_TABLE: dict[str, dict] = {
    _name: _record for _name, _record in _COMMANDS['commands'].items()
    if _name.lower() not in _MISCOLLECTED_COMMANDS
}

#: Member kinds that reflection reports and that the historical views expose. Fields and every
#: Extended Type System member (`ets_*`) are collected but withheld from these views, because the
#: prior database never saw them and the transforms that read the views were written against a
#: reflection-only member set. The query API exposes them for the migration that consumes them.
_VIEW_MEMBER_KINDS = frozenset({'method', 'property'})


def _view_members(record: dict) -> dict[str, dict]:
    return {
        name: member
        for name, member in record['members'].items()
        if member['source'] in ('reflection', 'wmi') and member['kind'] in _VIEW_MEMBER_KINDS
    }


TYPE_MEMBERS: dict[str, list[str]] = {}

for _full, _record in _TYPE_TABLE.items():
    if _record['kind'] == 'enum':
        _names = sorted(_record['enum_values'] or {})
    else:
        _names = sorted(_view_members(_record))
    TYPE_MEMBERS[_full.lower()] = _names

PROPERTY_TYPES: dict[tuple[str, str], str] = {}

for _full, _record in _TYPE_TABLE.items():
    if _record['kind'] == 'enum':
        continue
    _tl = _full.lower()
    for _name, _member in _view_members(_record).items():
        if _member['kind'] == 'property':
            PROPERTY_TYPES[(_tl, _name.lower())] = _member['type'].lower()

VARIABLE_TYPES: dict[str, str] = {
    _name.lower(): _info['type'].lower()
    for _name, _info in _VARIABLES['variables'].items()
    if _info['type'] is not None
}
#: `$PSCmdlet` exists only inside an advanced function's scope, so the pristine `Get-Variable` the
#: generator runs never sees it. It is supplied here, as the previous database did, because it
#: cannot be collected rather than because it is absent.
VARIABLE_TYPES.setdefault('pscmdlet', 'system.management.automation.psscriptcmdlet')

TYPE_ALIASES: dict[str, str] = {
    _alias.lower(): _full.lower() for _alias, _full in _ACCELERATORS.items()
}

#: The set of type-accelerator spellings, lowercased. An accelerator is already the shortest
#: readable name for its type, so display normalization leaves it as written rather than expanding
#: it to the verbose full name: `[ref]` and `[int]` stay, where `[System.Int32]` folds to `[Int32]`.
TYPE_ACCELERATORS: frozenset[str] = frozenset(_alias.lower() for _alias in _ACCELERATORS)

CANONICAL_TYPE_NAMES: dict[str, str] = {}

for _alias, _full in _ACCELERATORS.items():
    _display = _full.removeprefix('System.')
    CANONICAL_TYPE_NAMES[_alias.lower()] = _display
    CANONICAL_TYPE_NAMES[_full.lower()] = _display
for _full in _TYPE_TABLE:
    _display = _full.removeprefix('System.')
    CANONICAL_TYPE_NAMES.setdefault(_full.lower(), _display)
    CANONICAL_TYPE_NAMES.setdefault(_full.lower().removeprefix('system.'), _display)

MEMBER_LOOKUP: dict[str, dict[str, str]] = {}

for _type_lower, _members in TYPE_MEMBERS.items():
    MEMBER_LOOKUP[_type_lower] = {m.lower(): m for m in _members}

WMI_CLASS_NAMES: dict[str, str] = {}

for _namespace, _classes in _WMI['namespaces'].items():
    for _cls, _cls_info in _classes.items():
        _cls_lower = _cls.lower()
        WMI_CLASS_NAMES.setdefault(_cls_lower, _cls)
        CANONICAL_TYPE_NAMES.setdefault(_cls_lower, _cls)
        _lookup = MEMBER_LOOKUP.setdefault(_cls_lower, {})
        for _prop in _cls_info['properties']:
            _lookup.setdefault(_prop.lower(), _prop)

CANONICAL_TYPE_NAMES.setdefault(
    'management.automation.sessionstateinternal',
    'Management.Automation.SessionStateInternal',
)


def _resolve_type_name(name: str) -> str | None:
    """
    Resolve a type name (as written in PowerShell) to its canonical lowercase full .NET name.
    Handles short names like 'String', qualified names like 'Net.WebClient', and full names like
    'System.Net.WebClient'.
    """
    lower = name.lower()
    if lower in TYPE_MEMBERS:
        return lower
    if lower in TYPE_ALIASES:
        return TYPE_ALIASES[lower]
    prefixed = F'system.{lower}'
    if prefixed in TYPE_MEMBERS:
        return prefixed
    return None


def is_type(name: str, canonical_lower: str) -> bool:
    """
    Check whether a type name (as written in PowerShell source) resolves to the given canonical
    lowercase .NET type name.
    """
    resolved = _resolve_type_name(name)
    return resolved == canonical_lower


#: The aliases the host binds, and nothing else. An entry here is not a harmless surplus: nothing in
#: ordinary name lookup beats an alias, so a name added here is taken away from whatever 5.1 would
#: have given it. `childitem`, `item`, `member` and `variable` were listed here once and are not
#: aliases at all — they are the engine's implicit `Get-` retry, which is a *last resort* reached
#: only once the alias, function and cmdlet tables have missed, so `function item { }` beats it. As
#: aliases they claimed the name before the function tier, and a call to the script's own function
#: was rewritten into a call to the cmdlet, deleting the body that ran. That rule lives in
#: `refinery.lib.scripts.ps1.ast.implicit_get_retry` instead, where it is a retry.
#:
#: `fhx` and `gerr` were listed too, and neither target is on the host: `Format-Hex` and `Get-Error`
#: were both measured absent from 5.1, so a bare `gerr` was rewritten into a name the script then
#: could not run, and the entry injected `Get-Error` as a cmdlet through the loop below on top of
#: that. `Format-Hex` had leaked into the collected command table on its own as well and is withheld
#: from it by `_MISCOLLECTED_COMMANDS` above.
#:
#: **A wrong record in either table is never merely surplus, because two things read them in a
#: direction it corrupts.** `refinery.lib.scripts.ps1.deobfuscation.wildcards` matches a wildcard
#: against `KNOWN_CMDLETS` and emits the unique hit as a command, which is what made a `Format-Hex`
#: record produce a call to a command the host cannot run — the `fhx` defect reached through the
#: other table. And `refinery.lib.scripts.ps1.ast.implicit_get_retry` refuses a retry for any name
#: a table claims, so a record for a command the host does not have suppresses a retry 5.1 performs.
#: The dash-free records were suspected of being that and are not: all 27 were measured on 5.1 and
#: every one resolves. Ten are the engine's own default session functions (`cd..`, `cd\`, `help`,
#: `importsystemmodules`, `mkdir`, `more`, `oss`, `pause`, `prompt`, `tabexpansion2`), fifteen are
#: Pester, PSDesiredStateConfiguration and PSReadLine — all shipped with Windows and autoloaded by
#: command name, so 5.1 really does resolve `describe` and `configuration` on a stock box — and
#: `powershell` and `powershell_ise.exe` are programs, where suppressing the retry is the answer.
#:
#: Anything added to either table has to be measured on a host first, and the ps1 oracle corpus is
#: where such a measurement is recorded.
KNOWN_ALIAS: dict[str, str] = {
    _name.lower(): _definition for _name, _definition in _COMMANDS['aliases'].items()
}

#: The CimCmdlets short forms are module-provided aliases, and a bare pristine `Get-Alias` does not
#: list them even though their target cmdlets are collected. `run-pwsh.ps1` now imports the modules
#: before enumerating aliases, but the shipped data predates that fix, so they are restored here
#: until the next regeneration; each target is a canonical name already in `KNOWN_CMDLETS`.
for _cim_alias, _cim_command in {
    'gcai' : 'Get-CimAssociatedInstance',
    'gcim' : 'Get-CimInstance',
    'gcls' : 'Get-CimClass',
    'gcms' : 'Get-CimSession',
    'icim' : 'Invoke-CimMethod',
    'ncim' : 'New-CimInstance',
    'ncms' : 'New-CimSession',
    'ncso' : 'New-CimSessionOption',
    'rcie' : 'Register-CimIndicationEvent',
    'rcim' : 'Remove-CimInstance',
    'rcms' : 'Remove-CimSession',
    'scim' : 'Set-CimInstance',
}.items():
    KNOWN_ALIAS.setdefault(_cim_alias, _cim_command)

KNOWN_PS_OPERATORS: dict[str, str] = {name.lower(): name for name in [
    '-As',
    '-BAnd',
    '-BNot',
    '-BOr',
    '-BXor',
    '-Contains',
    '-CReplace',
    '-Eq',
    '-GE',
    '-GT',
    '-In',
    '-IReplace',
    '-Is',
    '-IsNot',
    '-Join',
    '-LE',
    '-Like',
    '-LT',
    '-Match',
    '-NE',
    '-Not',
    '-NotContains',
    '-NotIn',
    '-NotLike',
    '-NotMatch',
    '-Replace',
    '-Shl',
    '-Shr',
    '-Split',
    '-XOr',
]}

KNOWN_PS_SWITCHES: dict[str, str] = {name.lower(): name for name in [
    '-Command',
    '-EncodedCommand',
    '-Exec Bypass',
    '-ExecutionPolicy',
    '-File',
    '-InputFormat',
    '-NoExit',
    '-NoLogo',
    '-NoProfile',
    '-NonInter',
    '-OutputFormat',
    '-Sta',
    '-Version',
    '-Windows Hidden',
    '-WindowStyle',
]}

KNOWN_CMDLETS: dict[str, str] = {name.lower(): name for name in _COMMAND_TABLE}
KNOWN_CMDLETS.setdefault('convertfrom-base64', 'ConvertFrom-Base64')
KNOWN_CMDLETS.setdefault('powershell', 'PowerShell')

for _n in KNOWN_ALIAS.values():
    KNOWN_CMDLETS.setdefault(_n.lower(), _n)

CMDLET_PARAMETERS: dict[str, list[str]] = {
    _name.lower(): [
        _param for _param, _info in _record['parameters'].items() if not _info['common']
    ]
    for _name, _record in _COMMAND_TABLE.items()
}

ALL_PARAMETER_NAMES: dict[str, str] = {}

for _params in CMDLET_PARAMETERS.values():
    for _p in _params:
        ALL_PARAMETER_NAMES.setdefault(_p.lower(), _p)

#: The PowerShell common parameters, keyed by lowercased name and mapped to their lowercased
#: aliases. These are the parameters every advanced command shares (`-ErrorAction`, `-OutVariable`,
#: `-Verbose`, ...), which `CMDLET_PARAMETERS` and the views built from it deliberately exclude.
#: This is the one place they are surfaced, so a consumer reasoning about them — the out-variable
#: purity check does — reads them from the collected data rather than hardcoding the set. They are
#: identical on every command, so a single advanced command already determines the whole set; the
#: union is taken regardless so the first command that happens to lack one does not drop it.
COMMON_PARAMETERS: dict[str, tuple[str, ...]] = {}

for _record in _COMMAND_TABLE.values():
    for _param, _info in _record['parameters'].items():
        if _info['common']:
            COMMON_PARAMETERS.setdefault(
                _param.lower(),
                tuple(_alias.lower() for _alias in _info['aliases']),
            )

#: The common parameters that bind their argument as the *name* of a variable the command fills:
#: `-OutVariable`, `-ErrorVariable` and the rest, with their aliases (`ov`, `ev`, ...). A common
#: parameter names a variable exactly when its name ends in `Variable`, the convention the engine
#: defines them under, which `-OutBuffer` (a count) is the one common parameter to fail.
#:
#: These are the names a caller means when it asks which parameters address a variable by string,
#: and they are deliberately *not* the same set as the parameters that make a command impure:
#: `refinery.lib.scripts.ps1.analysis.effects` adds `-SetSeed` to that one, a `Get-Random` switch
#: that rewrites the generator state and names no variable at all. A consumer reading the impurity
#: set for names would take the `5` of `Get-Random -SetSeed 5` for a variable called `5`.
#: The out-variable parameters the derivation below must produce. Each is a fixed engine contract
#: whose loss silences a real write, so a collected surface that no longer carries one fails the
#: load rather than letting the set shrink silently.
REQUIRED_OUT_VARIABLE_PARAMETERS = frozenset({
    'errorvariable',
    'informationvariable',
    'outvariable',
    'pipelinevariable',
    'warningvariable',
})


def _derive_out_variable_parameters(common: dict[str, tuple[str, ...]]) -> frozenset[str]:
    """
    The out-variable parameters and their aliases, from a collected common-parameter surface.

    Taken as an argument rather than read from the module so the floor below can be exercised
    against a surface that has lost one; a check that only ever runs on the real data, at import,
    cannot be shown to work.
    """
    names: set[str] = set()
    for parameter, aliases in common.items():
        if parameter.endswith('variable'):
            names.add(parameter)
            names.update(aliases)
    if missing := REQUIRED_OUT_VARIABLE_PARAMETERS - names:
        raise ValueError(
            F'the collected common parameters no longer surface the out-variable parameters '
            F'{sorted(missing)!r}; every view built on them would silently stop treating those '
            F'parameters as naming a variable, so the data and this module are out of step.'
        )
    return frozenset(names)


OUT_VARIABLE_PARAMETERS = _derive_out_variable_parameters(COMMON_PARAMETERS)

_VALUE_PARAMETERS: dict[str, frozenset[str]] = {}

_COMMAND_RECORDS: dict[str, dict] = {
    _name.lower(): _record for _name, _record in _COMMAND_TABLE.items()
}


def value_parameters(command: str) -> frozenset[str]:
    """
    The lowercased parameter names and aliases of *command* that take a value, as opposed to the
    switches that stand alone. Empty for a command the collected surface does not carry.

    A caller reading a command's arguments needs this to tell `-Name x`, where `x` is the value of
    `-Name`, from `-Recurse C:\\`, where the path is a positional argument of its own. The parser
    renders both as a switch followed by a positional because it has no parameter metadata; this is
    that metadata.

    Looked up and memoized per command rather than built for every command at import: the table
    carries thousands of commands and a caller asks about a handful.
    """
    command = command.lower()
    found = _VALUE_PARAMETERS.get(command)
    if found is None:
        names: set[str] = set()
        for _parameter, _info in _COMMAND_RECORDS.get(command, {}).get('parameters', {}).items():
            if _info['switch']:
                continue
            names.add(_parameter.lower())
            names.update(_alias.lower() for _alias in _info['aliases'])
        found = _VALUE_PARAMETERS[command] = frozenset(names)
    return found


SIMPLE_IDENTIFIER = re.compile(r'^[a-zA-Z_]\w*$')

OBJ_COMMANDS = frozenset({
    'new-object',
})

WMI_COMMANDS = frozenset({
    'get-ciminstance',
    'get-wmiobject',
})

TYPE_ARG_COMMANDS = frozenset(OBJ_COMMANDS | WMI_COMMANDS)

GET_MEMBER_ALIASES = frozenset({'get-member', 'gm'})
GET_COMMAND_ALIASES = frozenset({'get-command', 'gcm'})

FOREACH_ALIASES = frozenset({'%', 'foreach', 'foreach-object'})

COMPARISON_OPS = {
    '-eq': operator.eq,
    '-ne': operator.ne,
    '-lt': operator.lt,
    '-le': operator.le,
    '-gt': operator.gt,
    '-ge': operator.ge,
}

ENCODING_MAP = {
    'ascii'            : 'ascii',            # noqa
    'bigendianunicode' : 'utf-16-be',        # noqa
    'default'          : 'latin-1',          # noqa
    'unicode'          : 'utf-16-le',        # noqa
    'utf7'             : 'utf-7',            # noqa
    'utf8'             : 'utf-8',            # noqa
    'utf32'            : 'utf-32-le',        # noqa
}

BUILTIN_VARIABLES = frozenset({'null', 'true', 'false'})

PS1_KNOWN_VARIABLES: dict[str, str] = {
    name.lower(): name for name in [
        'ConfirmPreference',
        'ConsoleFileName',
        'DebugPreference',
        'Error',
        'ErrorActionPreference',
        'ExecutionContext',
        'False',
        'ForEach',
        'FormatEnumerationLimit',
        'HOME',
        'Host',
        'InformationPreference',
        'Input',
        'Matches',
        'MaximumAliasCount',
        'MaximumDriveCount',
        'MaximumErrorCount',
        'MaximumFunctionCount',
        'MaximumHistoryCount',
        'MaximumVariableCount',
        'MyInvocation',
        'NestedPromptLevel',
        'Null',
        'OutputEncoding',
        'PID',
        'PROFILE',
        'ProgressPreference',
        'PSCommandPath',
        'PSCulture',
        'PSDefaultParameterValues',
        'PSEmailServer',
        'PSHome',
        'PSScriptRoot',
        'PSSessionApplicationName',
        'PSSessionConfigurationName',
        'PSSessionOption',
        'PSUICulture',
        'PSVersionTable',
        'PWD',
        'ShellID',
        'StackTrace',
        'This',
        'True',
        'VerbosePreference',
        'WarningPreference',
        'WhatIfPreference',
    ]
}

FORMAT_PATTERN = re.compile(r'\{\{|\}\}|\{(\d+)(?:,(-?\d+))?(?::([^}]+))?\}')


def resolve_type(name: str | Ps1TypeName) -> Ps1TypeName | None:
    """
    Resolve a .NET type name as written in PowerShell source to the one canonical `Ps1TypeName` the
    collected metadata is keyed by, or `None` when the name is syntactically not a type or names a
    type that was not collected. This understands the whole type-name grammar — accelerators, an
    omitted `System.` prefix, generic arity, arrays — and it is the **only** thing that mints a
    canonical name, which is what makes a canonical name comparable to another one.

    A parsed name is not canonical: `System.Int32`, `system.int32` and `System.Int32, mscorlib` parse
    to three unequal, differently hashing tuples, as do the two spellings of a list type. So the
    result is *rebuilt* rather than returned — the collected record's own casing for the name, the
    assembly qualification dropped because it does not distinguish a type here, and the arguments
    resolved in turn. An argument that does not resolve makes the whole name unresolved, since a
    name is only as understood as its least understood part.

    The array suffixes are carried rather than dropped. `char[]` used to resolve to `System.Char`,
    so every member question about an array of characters was answered off the element type; what
    an array's members actually are is `_member_surface`'s question, asked of the name this returns.
    """
    parsed = parse_type_name(name) if isinstance(name, str) else name
    if parsed is None:
        return None
    return _canonical_type_name(parsed)


def _canonical_type_name(parsed: Ps1TypeName) -> Ps1TypeName | None:
    for candidate in _definition_candidates(parsed.definition):
        if candidate not in _TYPE_TABLE and candidate not in _WMI_TYPES:
            continue
        arguments: list[Ps1TypeName] = []
        for argument in parsed.arguments:
            resolved = _canonical_type_name(argument)
            if resolved is None:
                return None
            arguments.append(resolved)
        base, _, arity = candidate.partition('`')
        return Ps1TypeName(
            name=base,
            arity=int(arity) if arity else 0,
            arguments=tuple(arguments),
            ranks=parsed.ranks,
            pointers=parsed.pointers,
            byref=parsed.byref,
        )
    return None


def _definition_candidates(definition: str):
    lower = definition.lower()
    accel = _ACCELERATORS.get(lower)
    if accel is not None:
        yield accel
    for key in _TYPE_LOOKUP.get(lower, ()):
        yield key


_TYPE_LOOKUP: dict[str, list[str]] = {}

for _full in _TYPE_TABLE:
    _TYPE_LOOKUP.setdefault(_full.lower(), []).append(_full)
    _bare = _full.removeprefix('System.').lower()
    if _bare != _full.lower():
        _TYPE_LOOKUP.setdefault(_bare, []).append(_full)

#: The WMI classes, shaped like a collected type record so that one resolver answers for them too. A
#: WMI class is a type a PowerShell expression can have — `Get-WmiObject Win32_Process` yields one —
#: and before this it was a lowercase string in a namespace of its own, which is one of the type
#: vocabularies this module exists to have only one of.
#:
#: The capture records a class's property *names* and not their types, so each member says so with a
#: `type` of `None`: a chain through a WMI property stops resolving there, which is the honest
#: answer rather than a guess. `source` is its own tier for the same reason `engine` is — a caller
#: deciding about purity must be able to tell a WMI property from a reflected one.
_WMI_TYPES: dict[str, dict] = {}

for _namespace, _classes in _WMI['namespaces'].items():
    for _cls, _cls_info in _classes.items():
        _WMI_TYPES.setdefault(_cls, {
            'kind': 'wmi',
            'sealed': False,
            'members': {
                _prop: {'kind': 'property', 'source': 'wmi', 'type': None}
                for _prop in _cls_info['properties']
            },
        })

for _cls in _WMI_TYPES:
    _TYPE_LOOKUP.setdefault(_cls.lower(), []).append(_cls)


def canonical_type(name: str) -> Ps1TypeName | None:
    """
    The canonical `Ps1TypeName` of the type a source name refers to, or `None` when it does not
    resolve to a collected type.
    """
    return resolve_type(name)


#: Where an array type's members come from. .NET gives every array the surface of `System.Array`
#: rather than one of its own, so `char[]` answers member questions off this and not off `Char`.
_ARRAY_SURFACE = 'System.Array'


def _type_record(key: str) -> dict | None:
    """
    The collected record a canonical definition key names, from either the .NET type capture or the
    WMI class capture. The two are separate captures of the same thing — what members a value of a
    type has — so they are read through one accessor and every query below is written once.
    """
    record = _TYPE_TABLE.get(key)
    if record is None:
        record = _WMI_TYPES.get(key)
    return record


def _member_surface(name: str | Ps1TypeName) -> str | None:
    """
    The key of the collected record whose members a value of this type carries, or `None` when the
    type does not resolve. This is the one place the array rule lives: an array carries the members
    of `System.Array` whatever it is an array *of*, so every member query — the table, the display
    order, one record, a property's type, the static overloads — reaches the same surface through
    here rather than each deciding for itself.

    Sealedness is deliberately not asked here, because it is not a question about the member
    surface: `System.Array` is not sealed, and every array type is.
    """
    resolved = resolve_type(name)
    if resolved is None:
        return None
    if resolved.ranks:
        return _ARRAY_SURFACE
    return resolved.definition


def type_members(name: str | Ps1TypeName) -> dict[str, dict] | None:
    """
    The full member table of a type, keyed by member name, including the fields and Extended Type
    System members the historical views omit. Each value carries at least `kind` and `source`.
    Returns `None` when the type is not collected.
    """
    key = _member_surface(name)
    if key is None:
        return None
    record = _type_record(key)
    return None if record is None else record['members']


def member_order(name: str | Ps1TypeName) -> list[str] | None:
    """
    The order `Get-Member` displays a type's members in, as observed on a real instance, or `None`
    when it was not collected for this type. This is the authentic display order, not a synthesis
    from the member table, and is only present for the types the generator has an instance for.
    """
    key = _member_surface(name)
    if key is None:
        return None
    record = _type_record(key)
    return None if record is None else record.get('member_order')


def type_is_sealed(name: str | Ps1TypeName) -> bool:
    """
    Whether the named type is sealed — no subtype of it exists, so a value of the type carries
    exactly the members reflection reports and nothing a subtype could add. `False` when the type
    is not collected or is not sealed, so a caller that needs sealedness to justify a grant fails
    closed on an unknown type rather than assuming it. The flag is read from the collected metadata,
    which is what retires the hand-asserted sealedness the effect layer's pure-read allow-list used
    to rest on.

    An array type is sealed whatever its element type is — .NET derives no type from one — so it is
    answered from the array rather than through `_member_surface`, which would answer it off
    `System.Array` and call it unsealed.
    """
    resolved = resolve_type(name)
    if resolved is None:
        return False
    if resolved.ranks:
        return True
    record = _type_record(resolved.definition)
    return record is not None and bool(record.get('sealed'))


class MemberLookup(enum.Enum):
    """
    The two non-record outcomes of `member_record`. A member query has three outcomes a purity gate
    must keep apart: the type was never collected, so nothing is known about its members; the type is
    collected but carries no member of that name; or the member is present and its record is returned.
    Collapsing the first two into a single `None` conflates an unknown surface with a known absence,
    which a sound gate must treat oppositely — an unknown surface is unsafe, a known-absent read
    yields `$null`.
    """
    UNCOLLECTED = 'uncollected'
    ABSENT = 'absent'


#: The members PowerShell's object adapter puts on every value, whatever its type. `Get-Member
#: -Force` reports them per instance rather than per type, so the capture — which walks types —
#: cannot hold them, and before this a read of one answered as though the member did not exist.
#:
#: Each is measured on a 5.1 host rather than reasoned about; see `TYPE_TRANSCRIPTS` in
#: `test.lib.scripts.ps1.test_oracle`. `Count` is 1 for a scalar, `PSTypeNames` is the type's own
#: name followed by its bases, and `PSObject` wraps the value.
#:
#: `Length` is deliberately absent even though a scalar answers 1 for it. Every type that carries a
#: real `Length` has it collected — `System.String`, `System.Array`, `System.IO.FileStream` — so the
#: tier would never be reached for one; putting it here would only add a way for the *next* type
#: whose capture is incomplete to have a live getter vouched for. The scalar case is left to the
#: value domain, which knows what a scalar is.
#:
#: `source` is neither `reflection` nor `ets`. Filing these as `ets` would make every read of one
#: impure and delete nothing that is deleted today; filing them as `reflection` would claim the
#: capture saw them. They are their own tier so that a gate can decide about them on purpose.
_ENGINE_MEMBERS: dict[str, dict] = {
    'count': {'kind': 'property', 'source': 'engine', 'type': 'System.Int32'},
    'pstypenames': {'kind': 'property', 'source': 'engine', 'type': 'System.String[]'},
    'psobject': {
        'kind': 'property',
        'source': 'engine',
        'type': 'System.Management.Automation.PSObject',
    },
}


def engine_member(member: str) -> dict | None:
    """
    The record for a member the object adapter adds to every value, or `None` for a name that is not
    one. Callers consult this only where `member_record` answered `MemberLookup.ABSENT`: a collected
    record is what the type really carries and is never overridden by this.
    """
    return _ENGINE_MEMBERS.get(member.lower())


def member_record(name: str | Ps1TypeName, member: str) -> dict | MemberLookup:
    """
    The collected record for a single member of a type, or a `MemberLookup` sentinel explaining why
    there is none. `MemberLookup.UNCOLLECTED` means the type has no member table at all;
    `MemberLookup.ABSENT` means the type is collected and carries no member of that name. The member
    is matched case-insensitively, as PowerShell resolves it, and the first match wins. The record
    carries at least `kind` and `source`, which distinguish a plain reflection property or field from
    a code-running Extended Type System member.

    A member the capture holds wins over `engine_member`, which is why the adapter tier is consulted
    here on the way out rather than by each caller: a caller that asked the tier first would answer
    `Count` off it for `System.Array`, which carries a real one.
    """
    members = type_members(name)
    if members is None:
        return MemberLookup.UNCOLLECTED
    lower = member.lower()
    for stored, record in members.items():
        if stored.lower() == lower:
            return record
    engine = engine_member(member)
    if engine is not None:
        return engine
    return MemberLookup.ABSENT


def view_members(name: str | Ps1TypeName) -> dict[str, dict] | None:
    """
    The members of a type that `Get-Member` reports without `-Force`: the reflected methods and
    properties, and for a WMI class its properties. Fields and Extended Type System members are
    withheld, as they are from the historical `TYPE_MEMBERS` view this replaces — the difference is
    that the type is resolved through `resolve_type`, so an array answers off `System.Array` and a
    spelling that is not already the lowercased `FullName` resolves instead of missing.

    An enum has no members in this sense and its named values stand in for them, which is what the
    view has always done and what a caller listing what may follow a dot needs.
    """
    key = _member_surface(name)
    if key is None:
        return None
    record = _type_record(key)
    if record is None:
        return None
    if record.get('kind') == 'enum':
        return {value: {'kind': 'property', 'source': 'enum'} for value in record['enum_values'] or {}}
    return _view_members(record)


def member_names(name: str | Ps1TypeName) -> list[str] | None:
    """
    The names `view_members` reports, sorted, or `None` when the type does not resolve.
    """
    members = view_members(name)
    return None if members is None else sorted(members)


def canonical_member(name: str | Ps1TypeName, member: str) -> str | None:
    """
    The casing the metadata records for a member, matched case-insensitively as PowerShell resolves
    it, or `None` when the type or the member does not resolve.
    """
    members = view_members(name)
    if members is None:
        return None
    lower = member.lower()
    for stored in members:
        if stored.lower() == lower:
            return stored
    return None


def resolve_member_type(name: str | Ps1TypeName, member: str) -> Ps1TypeName | None:
    """
    The type a property of a type holds, or `None` when the type does not resolve, carries no such
    member, or the member is not a property. The answer is a canonical `Ps1TypeName`, so a chain of
    reads composes: what `$x.Prop.Length` resolves to is this asked twice.
    """
    record = member_record(name, member)
    if isinstance(record, MemberLookup):
        return None
    if record.get('kind') != 'property':
        return None
    declared = record.get('type')
    if not declared:
        return None
    return resolve_type(declared)


def static_overloads(name: str | Ps1TypeName, member: str) -> list[dict]:
    """
    The static overloads of a method on a type, each a record carrying its `returns` and its
    `parameters`, where every parameter records its `byref`/`out` direction, its `type` and its
    `position`. Returns an empty list when the type is not collected or carries no static method of
    that name. The member is matched case-insensitively, as PowerShell resolves it, and instance
    overloads are excluded: a caller asks this to reason about a `[Type]::Member(...)` call, whose
    reachable surface is the static one. A non-method member that case-collides with the method name
    is skipped, not taken as the answer, so a real static method behind it is still found.
    """
    members = type_members(name)
    if members is None:
        return []
    for stored, record in members.items():
        if stored.lower() != member.lower():
            continue
        if record.get('kind') != 'method':
            continue
        return [overload for overload in record.get('overloads') or () if overload.get('static')]
    return []


_COMMAND_LOOKUP: dict[str, dict] = {
    _name.lower(): _record for _name, _record in _COMMAND_TABLE.items()
}


def command(name: str) -> dict | None:
    """
    The collected record for a command, or `None` when the name is not a known cmdlet or function.
    The record carries the command kind, its module, declared output types with an
    `output_type_declared` flag, and the full parameter table including the common parameters.
    Command names are matched case-insensitively, as PowerShell resolves them.
    """
    return _COMMAND_LOOKUP.get(name.lower())


def command_output_types(name: str) -> frozenset[str] | None:
    """
    The declared output types of a command, lowercased, or `None` when the command declares none. A
    set is returned only when the command carries an `[OutputType]` attribute — the
    `output_type_declared` flag — because an `output_types` list without it records what was observed,
    not what the author promised, and an empty one means the declaration was never made rather than
    that the command emits nothing. An unknown command is `None` for the same reason. Names are
    matched case-insensitively.
    """
    record = _COMMAND_LOOKUP.get(name.lower())
    if record is None or not record.get('output_type_declared'):
        return None
    return frozenset(_type.lower() for _type in record['output_types'])
