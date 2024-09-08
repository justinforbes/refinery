#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern matching based extraction and substitution units.
"""
import re

from typing import Iterable, Optional, Callable, Union, ByteString, Dict
from itertools import islice
from hashlib import blake2b

from refinery.lib.types import INF, AST, BufferOrStr
from refinery.lib.argformats import regexp
from refinery.units import Arg, Unit


class PatternExtractorBase(Unit, abstract=True):

    def __init__(
        self,
        min        : Arg.Number('-n', help='Matches must have length at least N.') = 1,
        max        : Arg.Number('-m', help='Matches must have length at most N.') = None,
        len        : Arg.Number('-e', help='Matches must be of length N.') = None,
        stripspace : Arg.Switch('-x', help='Strip all whitespace from input data.') = False,
        duplicates : Arg.Switch('-r', help='Yield every (transformed) Match, even when it was found before.') = False,
        longest    : Arg.Switch('-l', help='Sort results by length.') = False,
        take       : Arg.Number('-t', help='Return only the first N occurrences in order of appearance.') = None,
        **keywords
    ):
        keywords.setdefault('ascii', True)
        keywords.setdefault('utf16', True)
        super().__init__(
            min=min,
            max=max or INF,
            len=len or AST,
            stripspace=stripspace,
            duplicates=duplicates,
            longest=longest,
            take=take or INF,
            **keywords
        )

    def matches(self, data: ByteString, pattern: Union[ByteString, re.Pattern]):
        """
        Searches the input data for the given regular expression pattern. If the
        argument `utf16` is `True`, search for occurrences where a zero byte
        is between every character of the match. The `ascii` option allows to
        control whether normal matching results are also returned.
        """
        if not isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        if self.args.ascii:
            yield from pattern.finditer(data)
        if self.args.utf16:
            for zm in re.findall(BR'(?:.\0)+', data, flags=re.DOTALL):
                yield from pattern.finditer(zm[::2])

    def _prefilter(self, matches: Iterable[re.Match]) -> Iterable[re.Match]:
        barrier = set()
        taken = 0
        for match in matches:
            hit = memoryview(match[0])
            if not hit or len(hit) != self.args.len or len(hit) < self.args.min or len(hit) > self.args.max:
                continue
            if not self.args.duplicates:
                uid = int.from_bytes(blake2b(hit, digest_size=8).digest(), 'big')
                if uid in barrier:
                    continue
                barrier.add(uid)
            yield match
            taken += 1
            if not self.args.longest and taken >= self.args.take:
                break

    def _postfilter(self, matches: Iterable[re.Match]) -> Iterable[re.Match]:
        result = matches
        if self.args.longest and self.args.take and self.args.take is not INF:
            try:
                length = len(result)
            except TypeError:
                result = list(result)
                length = len(result)
            indices = sorted(range(length), key=lambda k: len(result[k][0]), reverse=True)
            for k in sorted(islice(indices, abs(self.args.take))):
                yield result[k]
        elif self.args.longest:
            yield from sorted(result, key=lambda m: m.end() - m.start(), reverse=True)
        elif self.args.take:
            yield from islice(result, abs(self.args.take))

    def matchfilter(self, matches: Iterable[re.Match]) -> Iterable[re.Match]:
        yield from self._postfilter(self._prefilter(matches))

    def matches_filtered(
        self,
        data: ByteString,
        pattern: Union[ByteString, re.Pattern],
        *transforms: Optional[Iterable[Callable[[re.Match], Optional[Union[Dict, ByteString]]]]]
    ) -> Iterable[Union[Dict, ByteString]]:
        """
        This is a wrapper for `AbstractRegexUint.matches` which filters the
        results according to the given commandline arguments. Returns a
        dictionary mapping its position (start, end) in the input data to the
        filtered and transformed match that was found at this position.
        """
        def funcify(t):
            def const(m): return t
            return t if callable(t) else const

        transforms = [funcify(f) for f in transforms] or [lambda m: m[0]]

        if self.args.stripspace:
            data = re.sub(BR'\s+', B'', data)
        for k, match in enumerate(self.matchfilter(self.matches(memoryview(data), pattern))):
            for transform in transforms:
                t = transform(match)
                if t is None:
                    continue
                t = self.labelled(t)
                t.set_next_batch(k)
                yield t


class PatternExtractor(PatternExtractorBase, abstract=True):
    def __init__(
        self, min=1, max=None, len=None, stripspace=False, duplicates=False, longest=False, take=None,
        ascii: Arg.Switch('-u', '--no-ascii', group='AvsU', help='Search for UTF16 encoded patterns only.') = True,
        utf16: Arg.Switch('-a', '--no-utf16', group='AvsU', help='Search for ASCII encoded patterns only.') = True,
        **keywords
    ):
        super().__init__(
            min=min,
            max=max,
            len=len,
            stripspace=stripspace,
            duplicates=duplicates,
            longest=longest,
            take=take,
            ascii=ascii,
            utf16=utf16,
            **keywords
        )


class RegexUnit(Unit, abstract=True):

    def __init__(
        self,
        fullmatch: Arg.Switch('-U', help=(
            'Regular expressions are matched against the full input, not substrings of it.')) = False,
        multiline: Arg.Switch('-M', help=(
            'Caret and dollar in regular expressions match the beginning and end of a line and '
            'a dot does not match line breaks.')) = False,
        ignorecase: Arg.Switch('-I', help=(
            'Ignore capitalization for alphabetic characters in regular expressions.')) = False,
        **keywords
    ):
        flags = re.MULTILINE if multiline else re.DOTALL
        if ignorecase:
            flags |= re.IGNORECASE
        super().__init__(flags=flags, fullmatch=fullmatch, **keywords)

    def _make_matcher(self, pattern: Optional[BufferOrStr], default=None):
        if pattern is None:
            return default
        if self.args.fullmatch:
            return self._make_regex(pattern).fullmatch
        else:
            return self._make_regex(pattern).search

    def _make_regex(self, pattern: Optional[BufferOrStr]):
        if pattern is None:
            return None
        if isinstance(pattern, str):
            pattern = pattern.encode(self.codec)
        elif not isinstance(pattern, bytes):
            pattern = bytes(pattern)
        return re.compile(pattern, flags=self.args.flags)


class SingleRegexUnit(RegexUnit, abstract=True):

    def __init__(
        self, regex: Arg(type=regexp, help='Regular expression to match.'),
        count: Arg.Number('-c', help='Specify the maximum number of operations to perform.') = 0,
        fullmatch=False, multiline=False, ignorecase=False, **keywords
    ):
        super().__init__(
            regex=regex,
            count=count,
            fullmatch=fullmatch,
            multiline=multiline,
            ignorecase=ignorecase,
            **keywords
        )

    @property
    def regex(self):
        return self._make_regex(self.args.regex)

    @property
    def matcher(self):
        return self._make_matcher(self.args.regex)
