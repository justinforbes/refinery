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

import gzip
import json
import operator
import re

from refinery.lib.resources import datapath
from refinery.lib.scripts.ps1.dotnet import parse_type_name

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

_ACCELERATORS: dict[str, str] = {
    _alias.lower(): _full for _alias, _full in _TYPES['accelerators'].items()
}
_TYPE_TABLE: dict[str, dict] = _TYPES['types']
_COMMAND_TABLE: dict[str, dict] = _COMMANDS['commands']

#: Member kinds that reflection reports and that the historical views expose. Fields and every
#: Extended Type System member (`ets_*`) are collected but withheld from these views, because the
#: prior database never saw them and the transforms that read the views were written against a
#: reflection-only member set. The query API exposes them for the migration that consumes them.
_VIEW_MEMBER_KINDS = frozenset({'method', 'property'})


def _view_members(record: dict) -> dict[str, dict]:
    return {
        name: member
        for name, member in record['members'].items()
        if member['source'] == 'reflection' and member['kind'] in _VIEW_MEMBER_KINDS
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


KNOWN_ALIAS: dict[str, str] = {
    _name.lower(): _definition for _name, _definition in _COMMANDS['aliases'].items()
}
KNOWN_ALIAS.setdefault('childitem', 'Get-ChildItem')
KNOWN_ALIAS.setdefault('fhx', 'Format-Hex')
KNOWN_ALIAS.setdefault('gerr', 'Get-Error')
KNOWN_ALIAS.setdefault('item', 'Get-Item')
KNOWN_ALIAS.setdefault('member', 'Get-Member')
KNOWN_ALIAS.setdefault('variable', 'Get-Variable')

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


def resolve_type(name: str) -> str | None:
    """
    Resolve a .NET type name as written in PowerShell source to the canonical key of the collected
    type record: the reflection `FullName` of the generic definition, e.g. `System.Int32` for `int`
    or `System.Collections.Generic.List` `` `1 `` for `[Collections.Generic.List[string]]`. Returns
    `None` when the name is syntactically not a type or names a type that was not collected.

    Unlike `_resolve_type_name`, which the historical views expose in lowercase, this understands the
    full type-name grammar — accelerators, an omitted `System.` prefix, generic arity, arrays — and
    returns the collected record's own casing.
    """
    parsed = parse_type_name(name)
    if parsed is None:
        return None
    for candidate in _definition_candidates(parsed.definition):
        if candidate in _TYPE_TABLE:
            return candidate
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


def canonical_type(name: str) -> str | None:
    """
    The canonical `FullName` of the type a source name refers to, or `None` when it does not resolve
    to a collected type.
    """
    return resolve_type(name)


def type_members(name: str) -> dict[str, dict] | None:
    """
    The full member table of a type, keyed by member name, including the fields and Extended Type
    System members the historical views omit. Each value carries at least `kind` and `source`.
    Returns `None` when the type is not collected.
    """
    key = resolve_type(name)
    if key is None:
        return None
    return _TYPE_TABLE[key]['members']


def member_order(name: str) -> list[str] | None:
    """
    The order `Get-Member` displays a type's members in, as observed on a real instance, or `None`
    when it was not collected for this type. This is the authentic display order, not a synthesis
    from the member table, and is only present for the types the generator has an instance for.
    """
    key = resolve_type(name)
    if key is None:
        return None
    return _TYPE_TABLE[key].get('member_order')


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
