import base64
import json

from refinery.lib.dns import (
    DnsClass,
    DnsHeaderFlag,
    DnsMessage,
    DnsMX,
    DnsOpcode,
    DnsQr,
    DnsQuestion,
    DnsRcode,
    DnsSOA,
    DnsSRV,
    DnsType,
    parse_dns_message,
)

from .. import TestUnitBase


_DNS_A_QUERY = base64.b85decode(
    '5;Oq-00961000002W5C+ZE$R517mM(000310R'
)

_DNS_A_RESPONSE = base64.b85decode(
    '5;TQ?00962000002W5C+ZE$R517mM(000310l*9Z0RRC200Arj1YNk;A^'
)

_DNS_CNAME_RESPONSE = base64.b85decode(
    'R(OGc009630000019x|K2W5C+ZE$R517mM(000310l*9Z1pom601glU0>BW!5C8!H0RR92EC2*uxY!~'
)

_DNS_MX_RESPONSE = base64.b85decode(
    'tIdIc00963000002W5C+ZE$R517mM(000jF0l*9Z4*&rG01glU2>=QNZDDC_zzo0)01p5G000gU015yU1#MwzY%;(M'
)

_DNS_TXT_RESPONSE = base64.b85decode(
    'zVCs6009620000026T09Ze?r)Wp-(717mM(000mG0l*9Z5C8!H002Ay5fD>HR&!!UIA>NeIeB77QD^'
)

_DNS_TXT_MULTI_RESPONSE = base64.b85decode(
    '%Km|X00962000001#NX~bZG>1Wpi``V{dH$01yBHzzhHo009610C)fs1!ie-b94rCWn*t{WCe6+X>w!'
)

_DNS_SOA_RESPONSE = base64.b85decode(
    'nVE%v00962000002W5C+ZE$R517mM(000I60l*9Z1^@v70a1VeA_H!7F~AH3VPtJ-Zomw9qw#eB01glU00V>o2|9oP0a1V'
)

_DNS_AAAA_RESPONSE = base64.b85decode(
    'Lm`2H00962000001Zi+~HV0*RVQp}1WdmbxZ2$lq00F=Z02}}T0003j01zMn4Y&XR0000000000009'
)

_DNS_SRV_RESPONSE = base64.b85decode(
    'cXxq+00962000001YdJ$a0Fj;V{iv$cwudDY-IyuZ*2eoApilu3;-bj0RR9F5C9hd1pojO!~=6_a0g|0VQp}1WdmbxZ2$'
)

_DNS_NS_RESPONSE = base64.b85decode(
    '5fO!e00963009612W5C+ZE$R517mM(000620l*9Z0ssL30a1Ve1_N$$F~AJK3;+TE0RRC}fB*&qZgVog48RNk1^@v70a1Ve'
    '9>6IDVPtJ-Zomw9qw#eB01glU00V>o2|9oP0a1V'
)


class TestDNS(TestUnitBase):

    def test_a_query(self):
        result = parse_dns_message(_DNS_A_QUERY)
        self.assertEqual(result.id, 0x1234)
        self.assertEqual(result.qr, DnsQr.QUERY)
        self.assertEqual(result.opcode, DnsOpcode.QUERY)
        self.assertEqual(result.flags, DnsHeaderFlag.RD)
        self.assertEqual(result.rcode, DnsRcode.NOERROR)
        self.assertEqual(result.question, [DnsQuestion('example.com', DnsType.A, DnsClass.IN)])
        self.assertEqual(result.answer, [])

    def test_a_response(self):
        result = parse_dns_message(_DNS_A_RESPONSE)
        self.assertEqual(result.id, 0x1234)
        self.assertEqual(result.qr, DnsQr.RESPONSE)
        self.assertEqual(result.flags, DnsHeaderFlag.AA | DnsHeaderFlag.RD | DnsHeaderFlag.RA)
        self.assertEqual(len(result.answer), 1)
        answer = result.answer[0]
        self.assertEqual(answer.name, 'example.com')
        self.assertEqual(answer.type, DnsType.A)
        self.assertEqual(answer.ttl, 300)
        self.assertEqual(answer.data, '93.184.216.34')

    def test_cname_with_compression(self):
        result = parse_dns_message(_DNS_CNAME_RESPONSE)
        self.assertEqual(len(result.answer), 2)
        self.assertEqual(result.answer[0].type, DnsType.CNAME)
        self.assertEqual(result.answer[0].data, 'example.com')
        self.assertEqual(result.answer[1].type, DnsType.A)
        self.assertEqual(result.answer[1].data, '93.184.216.34')

    def test_mx_response(self):
        result = parse_dns_message(_DNS_MX_RESPONSE)
        self.assertEqual(len(result.answer), 2)
        self.assertEqual(result.answer[0].data, DnsMX(10, 'mail.example.com'))
        self.assertEqual(result.answer[1].data, DnsMX(20, 'mail2.example.com'))

    def test_txt_single_string(self):
        result = parse_dns_message(_DNS_TXT_RESPONSE)
        self.assertEqual(result.answer[0].data, ['SGVsbG8gV29ybGQh'])

    def test_txt_multiple_strings(self):
        result = parse_dns_message(_DNS_TXT_MULTI_RESPONSE)
        self.assertEqual(result.answer[0].data, ['first', 'second', 'third'])

    def test_soa_response(self):
        result = parse_dns_message(_DNS_SOA_RESPONSE)
        self.assertEqual(result.answer[0].data, DnsSOA(
            mname='ns1.example.com',
            rname='admin.example.com',
            serial=2024010101,
            refresh=3600,
            retry=900,
            expire=604800,
            minimum=86400,
        ))

    def test_aaaa_response(self):
        result = parse_dns_message(_DNS_AAAA_RESPONSE)
        self.assertEqual(result.answer[0].data, '2001:db8::1')

    def test_srv_response(self):
        result = parse_dns_message(_DNS_SRV_RESPONSE)
        self.assertEqual(result.answer[0].data, DnsSRV(5, 0, 5060, 'sip.example.com'))

    def test_ns_with_authority_section(self):
        result = parse_dns_message(_DNS_NS_RESPONSE)
        self.assertEqual(len(result.answer), 2)
        self.assertEqual(result.answer[0].data, 'ns1.example.com')
        self.assertEqual(result.answer[1].data, 'ns2.example.com')
        self.assertEqual(result.authority[0].type, DnsType.SOA)

    def test_empty_sections_are_empty_lists(self):
        result = parse_dns_message(_DNS_A_QUERY)
        self.assertEqual(result.answer, [])
        self.assertEqual(result.authority, [])
        self.assertEqual(result.additional, [])

    def test_truncated_message_does_not_crash(self):
        for n in range(len(_DNS_A_RESPONSE)):
            result = parse_dns_message(_DNS_A_RESPONSE[:n])
            self.assertTrue(result is None or isinstance(result, DnsMessage))

    def test_circular_pointer_does_not_loop(self):
        crafted = bytearray(_DNS_A_RESPONSE)
        crafted[12] = 0xC0
        crafted[13] = 12
        result = parse_dns_message(crafted)
        self.assertIsInstance(result, DnsMessage)

    def test_handles_accepts_valid_dns(self):
        unit = self.load()
        self.assertTrue(unit.handles(_DNS_A_QUERY))
        self.assertTrue(unit.handles(_DNS_A_RESPONSE))

    def test_handles_rejects_non_dns(self):
        unit = self.load()
        self.assertIsNone(unit.handles(b'This is not DNS'))
        self.assertIsNone(unit.handles(b'\x00' * 5))

    def test_unit_json_query(self):
        result = json.loads(_DNS_A_QUERY | self.load() | bytes)
        self.assertEqual(result, {
            'id'       : 0x1234,
            'qr'       : 'query',
            'opcode'   : 'QUERY',
            'flags'    : ['rd'],
            'rcode'    : 'NOERROR',
            'question' : [{'name': 'example.com', 'type': 'A', 'class': 'IN'}],
        })

    def test_unit_json_response_full_shape(self):
        result = json.loads(_DNS_A_RESPONSE | self.load() | bytes)
        self.assertEqual(result, {
            'id'       : 0x1234,
            'qr'       : 'response',
            'opcode'   : 'QUERY',
            'flags'    : ['aa', 'rd', 'ra'],
            'rcode'    : 'NOERROR',
            'question' : [{'name': 'example.com', 'type': 'A', 'class': 'IN'}],
            'answer'   : [{
                'name'  : 'example.com',
                'type'  : 'A',
                'class' : 'IN',
                'ttl'   : 300,
                'data'  : '93.184.216.34',
            }],
        })

    def test_unit_json_mx_renders_named_fields(self):
        result = json.loads(_DNS_MX_RESPONSE | self.load() | bytes)
        self.assertEqual(result['answer'][0]['data'], {'preference': 10, 'exchange': 'mail.example.com'})
