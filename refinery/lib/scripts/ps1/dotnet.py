"""
The grammar of .NET type names, as PowerShell source writes them and as .NET reflection reports
them. This module is purely syntactic: it turns the text of a type name into a `Ps1TypeName` and
back, and it knows nothing about which types exist. Deciding what a name refers to — resolving an
accelerator like `int`, supplying an omitted `System.` prefix, looking up members — requires the
collected metadata and belongs to `refinery.lib.scripts.ps1.data`.

The two spellings this has to bridge are the source form and the reflection form:

    System.Collections.Generic.List[int]
    System.Collections.Generic.List`1[[System.Int32, mscorlib]]

Both name the same open generic type, whose reflection `FullName` is the `Ps1TypeName.definition`
of either parse.
"""
from __future__ import annotations

import re

from typing import NamedTuple

_ARITY_SUFFIX = re.compile(r'`(\d+)$')

_IDENTIFIER = re.compile(
    r'''
    [^\W\d]\w* (?: `\d+ )?
    (?: [.+] [^\W\d]\w* (?: `\d+ )? )*
    ''',
    re.VERBOSE,
)


class Ps1TypeName(NamedTuple):
    """
    A parsed .NET type name.

    The `name` is the dotted base name with no generic arity marker, no type arguments and no array
    suffix; `System.Collections.Generic.List` for every spelling of a list type. `arity` is the
    number of generic parameters, which a name can carry without naming them — an open generic. When
    the arguments are named, `arguments` holds them and its length equals `arity`. `ranks` records
    the array suffixes in the order they were written, one entry per suffix, holding that suffix's
    dimension count, so `int[,][]` parses to `(2, 1)`. `pointers` counts trailing `*` suffixes and
    `byref` records a trailing `&`, which is how reflection reports the type of an `out` or `ref`
    parameter. `assembly` holds the assembly qualification that a reflection-form name may carry
    after a comma.

    The three suffix kinds are accepted only in the order the runtime writes them — arrays, then
    pointers, then at most one `&` — so a name that composes them differently is rejected rather
    than flattened onto fields that cannot express it.
    """

    name: str
    arity: int = 0
    arguments: tuple[Ps1TypeName, ...] = ()
    ranks: tuple[int, ...] = ()
    pointers: int = 0
    byref: bool = False
    assembly: str | None = None

    @property
    def definition(self) -> str:
        """
        The reflection `FullName` of the generic type definition, i.e. the base name carrying its
        arity marker but not its arguments. This is the form the collected type table is keyed by,
        because members are a property of the definition rather than of any one instantiation.
        """
        if self.arity:
            return F'{self.name}`{self.arity}'
        return self.name

    @property
    def generic_definition(self) -> Ps1TypeName:
        """
        This name with its type arguments dropped, which is the form anything keyed by what a type
        *has* rather than by what it *is* must use: members belong to the definition and not to any
        one instantiation, so `List[byte]` and `List[string]` carry the same surface and have to
        reach it through the same key. `definition` is the same idea as text; this is it as a name,
        so a table of names can be looked up in without spelling one.
        """
        if not self.arguments:
            return self
        return self._replace(arguments=())

    @property
    def is_array(self) -> bool:
        return bool(self.ranks)

    def __str__(self):
        """
        Render the name in PowerShell source form. The assembly qualification is deliberately not
        rendered: it has no source spelling outside the bracketed reflection form, where it would
        have to be re-bracketed to parse back.
        """
        text = self.name
        if self.arguments:
            arguments = ', '.join(str(argument) for argument in self.arguments)
            text = F'{text}[{arguments}]'
        elif self.arity:
            text = F'{text}`{self.arity}'
        for rank in self.ranks:
            commas = ',' * (rank - 1)
            text = F'{text}[{commas}]'
        text = text + '*' * self.pointers
        if self.byref:
            text = F'{text}&'
        return text


class _TypeNameSyntaxError(Exception):
    pass


class _TypeNameParser:

    def __init__(self, text: str):
        self._text = text
        self._pos = 0

    def parse(self) -> Ps1TypeName:
        result = self._parse_type(qualified=True)
        self._skip_space()
        if self._pos != len(self._text):
            raise _TypeNameSyntaxError
        return result

    def _skip_space(self):
        text = self._text
        while self._pos < len(text) and text[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        self._skip_space()
        if self._pos >= len(self._text):
            return ''
        return self._text[self._pos]

    def _parse_type(self, qualified: bool) -> Ps1TypeName:
        name, arity = self._parse_name()
        arguments: tuple[Ps1TypeName, ...] = ()
        if self._peek() == '[' and not self._at_rank_specifier():
            arguments = self._parse_arguments()
            if arity == 0:
                arity = len(arguments)
            elif arity != len(arguments):
                raise _TypeNameSyntaxError
        ranks: list[int] = []
        while self._peek() == '[':
            if not self._at_rank_specifier():
                raise _TypeNameSyntaxError
            ranks.append(self._parse_rank())
        pointers = 0
        while self._peek() == '*':
            pointers += 1
            self._pos += 1
        byref = self._peek() == '&'
        if byref:
            self._pos += 1
        if self._peek() in ('[', '*', '&'):
            raise _TypeNameSyntaxError
        assembly = None
        if qualified and self._peek() == ',':
            self._pos += 1
            assembly = self._parse_assembly()
        return Ps1TypeName(name, arity, arguments, tuple(ranks), pointers, byref, assembly)

    def _parse_name(self) -> tuple[str, int]:
        text = self._text
        parts: list[str] = []
        while self._pos < len(text):
            char = text[self._pos]
            if char.isspace():
                self._pos += 1
                continue
            if char in '[],*&':
                break
            parts.append(char)
            self._pos += 1
        name = ''.join(parts)
        if not _IDENTIFIER.fullmatch(name):
            raise _TypeNameSyntaxError
        arity = 0
        suffix = _ARITY_SUFFIX.search(name)
        if suffix is not None:
            arity = int(suffix[1])
            name = name[:suffix.start()]
        return name, arity

    def _parse_arguments(self) -> tuple[Ps1TypeName, ...]:
        self._pos += 1
        arguments: list[Ps1TypeName] = []
        while True:
            arguments.append(self._parse_argument())
            char = self._peek()
            self._pos += 1
            if char == ',':
                continue
            if char == ']':
                return tuple(arguments)
            raise _TypeNameSyntaxError

    def _parse_argument(self) -> Ps1TypeName:
        if self._peek() != '[':
            return self._parse_type(qualified=False)
        self._pos += 1
        argument = self._parse_type(qualified=True)
        if self._peek() != ']':
            raise _TypeNameSyntaxError
        self._pos += 1
        return argument

    def _at_rank_specifier(self) -> bool:
        """
        Whether the bracket at the cursor opens an array suffix rather than a generic argument list.
        The two are told apart by their contents: an array suffix holds only commas.
        """
        text = self._text
        cursor = self._pos + 1
        while cursor < len(text):
            char = text[cursor]
            if char == ']':
                return True
            if char != ',' and not char.isspace():
                return False
            cursor += 1
        return False

    def _parse_rank(self) -> int:
        self._pos += 1
        rank = 1
        while True:
            char = self._peek()
            self._pos += 1
            if char == ',':
                rank += 1
            elif char == ']':
                return rank
            else:
                raise _TypeNameSyntaxError

    def _parse_assembly(self) -> str:
        """
        Consume an assembly qualification, which runs to the bracket that closes the enclosing type
        argument or, at the outermost level, to the end of the name. It cannot stop at a comma
        because the qualification is itself a comma-separated list of assembly attributes.
        """
        text = self._text
        start = self._pos
        depth = 0
        while self._pos < len(text):
            char = text[self._pos]
            if char == '[':
                depth += 1
            elif char == ']':
                if depth == 0:
                    break
                depth -= 1
            self._pos += 1
        assembly = text[start:self._pos].strip()
        if not assembly:
            raise _TypeNameSyntaxError
        return assembly


def parse_type_name(text: str) -> Ps1TypeName | None:
    """
    Parse a .NET type name in either source or reflection form, returning `None` when the text is
    not one. The input is the name alone: the brackets of a PowerShell type literal are the literal's
    syntax rather than the name's, and the node model has already stripped them.

    Returning `None` rather than a best effort is the point. A caller that receives a `Ps1TypeName`
    has a name every part of which was understood, so a later lookup miss means the type is absent
    from the metadata — not that the name was mangled on the way in.
    """
    try:
        return _TypeNameParser(text).parse()
    except _TypeNameSyntaxError:
        return None
