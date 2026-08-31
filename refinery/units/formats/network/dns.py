from __future__ import annotations

from refinery.lib.dns import (
    DnsMX,
    DnsOpt,
    DnsRecord,
    DnsSOA,
    DnsSRV,
    DnsType,
    parse_dns_message,
)
from refinery.units.formats import JSONTableUnit


def _render_rdata(data):
    if isinstance(data, (DnsMX, DnsSOA, DnsSRV)):
        return data._asdict()
    return data


def _render_record(record: DnsRecord | DnsOpt) -> dict:
    if isinstance(record, DnsOpt):
        return {
            'type'           : DnsType.OPT.name,
            'udp_size'       : record.udp_size,
            'extended_rcode' : record.extended_rcode,
            'version'        : record.version,
            'flags'          : ['do'] if record.do else [],
        }
    return {
        'name'  : record.name,
        'type'  : record.type.name,
        'class' : record.cls.name,
        'ttl'   : record.ttl,
        'data'  : _render_rdata(record.data),
    }


class dns(JSONTableUnit):
    """
    Parse DNS wire-format messages and produce a JSON representation of the query or response.

    The intended usage is the pipeline `pcap [| udp | dns ]`, where `refinery.pcap` extracts
    network-layer packets, `refinery.udp` emits individual UDP datagrams, and this unit parses
    the DNS payload of each datagram. It can also be used standalone on raw DNS message bytes.
    """
    @classmethod
    def handles(cls, data) -> bool | None:
        if len(data) < 12:
            return None
        flags = int.from_bytes(data[2:4], 'big')
        opcode = (flags >> 11) & 0xF
        if opcode > 5:
            return None
        z_bit = (flags >> 6) & 1
        if z_bit:
            return None
        qdcount = int.from_bytes(data[4:6], 'big')
        if qdcount > 128:
            return None
        return True

    def json(self, data):
        message = parse_dns_message(data)
        if message is None:
            return None
        result = {
            'id'     : message.id,
            'qr'     : message.qr.name.lower(),
            'opcode' : message.opcode.name,
            'flags'  : [_n.lower() for flag in message.flags if (_n := flag.name) is not None],
            'rcode'  : message.rcode.name,
        }
        if message.question:
            result['question'] = [
                {'name': q.name, 'type': q.type.name, 'class': q.cls.name}
                for q in message.question
            ]
        if message.additional:
            result['additional'] = [_render_record(r) for r in message.additional]
        if message.answer:
            result['answer'] = [_render_record(r) for r in message.answer]
        if message.authority:
            result['authority'] = [_render_record(r) for r in message.authority]
        return result
