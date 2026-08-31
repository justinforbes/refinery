"""
DNS wire-format parser implementing RFC 1035 message parsing with support for name compression,
common record types, and EDNS(0). The main entry point is `parse_dns_message`, which returns a
`DnsMessage` built entirely from `enum.Enum` and `typing.NamedTuple` values rather than dictionaries.
"""
from __future__ import annotations

import ipaddress

from enum import IntEnum, IntFlag
from typing import NamedTuple

from refinery.lib.structures import EOF, StructReader


class DnsType(IntEnum):
    """
    DNS resource record type codes (RFC 1035, plus common extensions).
    """
    A     = 1  # noqa
    NS    = 2  # noqa
    CNAME = 5  # noqa
    SOA   = 6  # noqa
    PTR   = 12 # noqa
    MX    = 15 # noqa
    TXT   = 16 # noqa
    AAAA  = 28 # noqa
    SRV   = 33 # noqa
    OPT   = 41 # noqa

    @classmethod
    def _missing_(cls, value: object):
        obj = int.__new__(cls, int(value))  # type: ignore[arg-type]
        obj._name_ = F'TYPE{value}'
        obj._value_ = int(value)  # type: ignore[assignment]
        return obj


class DnsClass(IntEnum):
    """
    DNS class codes (RFC 1035).
    """
    IN  = 1  # noqa
    CS  = 2  # noqa
    CH  = 3  # noqa
    HS  = 4  # noqa
    ANY = 255

    @classmethod
    def _missing_(cls, value: object):
        obj = int.__new__(cls, int(value))  # type: ignore[arg-type]
        obj._name_ = F'CLASS{value}'
        obj._value_ = int(value)  # type: ignore[assignment]
        return obj


class DnsOpcode(IntEnum):
    """
    DNS opcode values from the header flags field.
    """
    QUERY  = 0  # noqa
    IQUERY = 1
    STATUS = 2
    NOTIFY = 4
    UPDATE = 5

    @classmethod
    def _missing_(cls, value: object):
        obj = int.__new__(cls, int(value))  # type: ignore[arg-type]
        obj._name_ = F'OPCODE{value}'
        obj._value_ = int(value)  # type: ignore[assignment]
        return obj


class DnsRcode(IntEnum):
    """
    DNS response code values.
    """
    NOERROR  = 0  # noqa
    FORMERR  = 1  # noqa
    SERVFAIL = 2  # noqa
    NXDOMAIN = 3  # noqa
    NOTIMP   = 4  # noqa
    REFUSED  = 5  # noqa
    YXDOMAIN = 6  # noqa
    YXRRSET  = 7  # noqa
    NXRRSET  = 8  # noqa
    NOTAUTH  = 9  # noqa
    NOTZONE  = 10 # noqa

    @classmethod
    def _missing_(cls, value: object):
        obj = int.__new__(cls, int(value))  # type: ignore[arg-type]
        obj._name_ = F'RCODE{value}'
        obj._value_ = int(value)  # type: ignore[assignment]
        return obj


class DnsQr(IntEnum):
    """
    The query/response bit from the header flags field.
    """
    QUERY    = 0  # noqa
    RESPONSE = 1


class DnsHeaderFlag(IntFlag):
    """
    The single-bit boolean flags carried in the DNS header flags field. Members are declared in the
    order in which they appear from the most significant bit so that iteration yields them in a
    stable, wire-consistent order.
    """
    AA = 1 << 10
    TC = 1 << 9
    RD = 1 << 8
    RA = 1 << 7
    AD = 1 << 5
    CD = 1 << 4


class DnsMX(NamedTuple):
    """
    The parsed payload of an `DnsType.MX` record.
    """
    preference: int
    exchange: str


class DnsSOA(NamedTuple):
    """
    The parsed payload of an `DnsType.SOA` record.
    """
    mname: str
    rname: str
    serial: int
    refresh: int
    retry: int
    expire: int
    minimum: int


class DnsSRV(NamedTuple):
    """
    The parsed payload of an `DnsType.SRV` record.
    """
    priority: int
    weight: int
    port: int
    target: str


class DnsOpt(NamedTuple):
    """
    The parsed EDNS(0) pseudo-record (`DnsType.OPT`). The class and TTL fields of the record header
    are repurposed by EDNS(0): the class carries the requestor's UDP payload size and the TTL field
    carries the extended response code, the EDNS version, and the DNSSEC OK flag.
    """
    udp_size: int
    extended_rcode: int
    version: int
    do: bool


DnsRData = str | list[str] | DnsMX | DnsSOA | DnsSRV
"""
The union of possible parsed record payloads. Address records resolve to a dotted string, name
records to the referenced domain, text records to a list of strings, and structured records to
their respective `typing.NamedTuple`. Unknown types fall back to a hexadecimal string.
"""


class DnsQuestion(NamedTuple):
    name: str
    type: DnsType
    cls: DnsClass


class DnsRecord(NamedTuple):
    name: str
    type: DnsType
    cls: DnsClass
    ttl: int
    data: DnsRData


class DnsMessage(NamedTuple):
    """
    A fully parsed DNS message. The four sections are always present as lists and may be empty.
    """
    id: int
    qr: DnsQr
    opcode: DnsOpcode
    flags: DnsHeaderFlag
    rcode: DnsRcode
    question: list[DnsQuestion]
    answer: list[DnsRecord | DnsOpt]
    authority: list[DnsRecord | DnsOpt]
    additional: list[DnsRecord | DnsOpt]


_MAX_NAME_LENGTH = 255
_MAX_SECTION_COUNT = 256
_HEADER_LENGTH = 12

_ALL_HEADER_FLAGS = (
    DnsHeaderFlag.AA
    | DnsHeaderFlag.TC
    | DnsHeaderFlag.RD
    | DnsHeaderFlag.RA
    | DnsHeaderFlag.AD
    | DnsHeaderFlag.CD
)


def _read_dns_name(msg: memoryview, start: int) -> tuple[str, int]:
    """
    Read a DNS domain name starting at the given offset within the full message buffer.
    Handles compression pointers (RFC 1035 section 4.1.4). Returns the dotted domain name and
    the number of bytes consumed from the starting offset (which may be less than the full
    resolved name due to pointer indirection).
    """
    labels: list[str] = []
    offset = start
    jumped = False
    bytes_consumed = 0
    visited: set[int] = set()
    total_length = 0

    while True:
        if offset >= len(msg):
            break
        if offset in visited:
            break
        visited.add(offset)

        length = msg[offset]

        if length == 0:
            if not jumped:
                bytes_consumed = offset - start + 1
            break
        elif (length & 0xC0) == 0xC0:
            if offset + 1 >= len(msg):
                if not jumped:
                    bytes_consumed = offset - start + 2
                break
            pointer = ((length & 0x3F) << 8) | msg[offset + 1]
            if not jumped:
                bytes_consumed = offset - start + 2
                jumped = True
            offset = pointer
        elif (length & 0xC0) != 0:
            if not jumped:
                bytes_consumed = offset - start + 1
            break
        else:
            offset += 1
            end = offset + length
            if end > len(msg):
                if not jumped:
                    bytes_consumed = end - start
                break
            total_length += length + 1
            if total_length > _MAX_NAME_LENGTH:
                break
            label = bytes(msg[offset:end])
            labels.append(label.decode('ascii', errors='replace'))
            offset = end

    if not bytes_consumed:
        bytes_consumed = offset - start

    return '.'.join(labels), bytes_consumed


def _parse_rdata(
    msg: memoryview, rtype: DnsType, rdata_offset: int, rdlength: int,
) -> DnsRData:
    available = len(msg) - rdata_offset
    if rdlength > available:
        rdlength = available
    rdata = msg[rdata_offset:rdata_offset + rdlength]
    if rtype == DnsType.A:
        if rdlength >= 4:
            return str(ipaddress.IPv4Address(bytes(rdata[:4])))
    elif rtype == DnsType.AAAA:
        if rdlength >= 16:
            return str(ipaddress.IPv6Address(bytes(rdata[:16])))
    elif rtype in (DnsType.CNAME, DnsType.NS, DnsType.PTR):
        name, _ = _read_dns_name(msg, rdata_offset)
        return name
    elif rtype == DnsType.MX:
        if rdlength >= 3:
            preference = int.from_bytes(rdata[:2], 'big')
            exchange, _ = _read_dns_name(msg, rdata_offset + 2)
            return DnsMX(preference, exchange)
    elif rtype == DnsType.SOA:
        mname, consumed = _read_dns_name(msg, rdata_offset)
        rname, consumed2 = _read_dns_name(msg, rdata_offset + consumed)
        numbers_start = consumed + consumed2
        if rdlength >= numbers_start + 20:
            tail = rdata[numbers_start:]
            return DnsSOA(
                mname=mname,
                rname=rname,
                serial=int.from_bytes(tail[0:4], 'big'),
                refresh=int.from_bytes(tail[4:8], 'big'),
                retry=int.from_bytes(tail[8:12], 'big'),
                expire=int.from_bytes(tail[12:16], 'big'),
                minimum=int.from_bytes(tail[16:20], 'big'),
            )
    elif rtype == DnsType.TXT:
        strings: list[str] = []
        pos = 0
        while pos < rdlength:
            length = rdata[pos]
            pos += 1
            end = min(pos + length, rdlength)
            strings.append(bytes(rdata[pos:end]).decode('latin-1'))
            pos = end
        return strings
    elif rtype == DnsType.SRV:
        if rdlength >= 7:
            priority = int.from_bytes(rdata[0:2], 'big')
            weight = int.from_bytes(rdata[2:4], 'big')
            port = int.from_bytes(rdata[4:6], 'big')
            target, _ = _read_dns_name(msg, rdata_offset + 6)
            return DnsSRV(priority, weight, port, target)
    return bytes(rdata).hex()


def _read_question(msg: memoryview, reader: StructReader) -> DnsQuestion:
    offset = reader.tell()
    name, consumed = _read_dns_name(msg, offset)
    reader.seekrel(consumed)
    qtype = DnsType(reader.u16())
    qclass = DnsClass(reader.u16())
    return DnsQuestion(name, qtype, qclass)


def _read_opt(reader: StructReader) -> DnsOpt:
    udp_size = reader.u16()
    ttl_field = reader.u32()
    rdlength = reader.u16()
    reader.seekrel(rdlength)
    return DnsOpt(
        udp_size=udp_size,
        extended_rcode=(ttl_field >> 24) & 0xFF,
        version=(ttl_field >> 16) & 0xFF,
        do=bool((ttl_field >> 15) & 1),
    )


def _read_record(msg: memoryview, reader: StructReader) -> DnsRecord | DnsOpt:
    offset = reader.tell()
    name, consumed = _read_dns_name(msg, offset)
    reader.seekrel(consumed)
    rtype = DnsType(reader.u16())
    if rtype == DnsType.OPT:
        return _read_opt(reader)
    rclass = DnsClass(reader.u16())
    ttl = reader.i32()
    rdlength = reader.u16()
    rdata_offset = reader.tell()
    reader.seekrel(rdlength)
    data = _parse_rdata(msg, rtype, rdata_offset, rdlength)
    return DnsRecord(name, rtype, rclass, ttl, data)


def _read_section(
    msg: memoryview, reader: StructReader, count: int,
) -> list[DnsRecord | DnsOpt]:
    records: list[DnsRecord | DnsOpt] = []
    for _ in range(min(count, _MAX_SECTION_COUNT)):
        try:
            records.append(_read_record(msg, reader))
        except EOF:
            break
    return records


def parse_dns_message(data: bytes | bytearray | memoryview) -> DnsMessage | None:
    """
    Parse a DNS wire-format message into a `DnsMessage`. Returns `None` if the data is too short to
    contain a valid DNS header.
    """
    msg = memoryview(data)
    if len(msg) < _HEADER_LENGTH:
        return None

    reader = StructReader(msg, bigendian=True)
    try:
        msg_id = reader.u16()
        flags_word = reader.u16()
        qdcount = reader.u16()
        ancount = reader.u16()
        nscount = reader.u16()
        arcount = reader.u16()
    except EOF:
        return None

    qr = DnsQr((flags_word >> 15) & 1)
    opcode = DnsOpcode((flags_word >> 11) & 0xF)
    rcode = DnsRcode(flags_word & 0xF)
    flags = DnsHeaderFlag(flags_word) & _ALL_HEADER_FLAGS

    questions: list[DnsQuestion] = []
    for _ in range(min(qdcount, _MAX_SECTION_COUNT)):
        try:
            questions.append(_read_question(msg, reader))
        except EOF:
            break

    return DnsMessage(
        id=msg_id,
        qr=qr,
        opcode=opcode,
        flags=flags,
        rcode=rcode,
        question=questions,
        answer=_read_section(msg, reader, ancount),
        authority=_read_section(msg, reader, nscount),
        additional=_read_section(msg, reader, arcount),
    )
