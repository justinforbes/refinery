from __future__ import annotations

from test import TestBase

from refinery.lib.scripts.js.deobfuscation import deobfuscate
from refinery.lib.scripts.js.deobfuscation.helpers import make_string_literal
from refinery.lib.scripts.js.parser import JsParser
from refinery.lib.scripts.js.synth import JsSynthesizer


class TestJsDeobfuscator(TestBase):

    def _deobfuscate(self, source: str) -> str:
        ast = JsParser(source).parse()
        deobfuscate(ast)
        return JsSynthesizer().convert(ast)


class TestBasicSimplifications(TestJsDeobfuscator):

    def _deobfuscate(self, source: str) -> str:
        ast = JsParser(source).parse()
        deobfuscate(ast)
        return JsSynthesizer().convert(ast)

    def test_string_concat_simple(self):
        result = self._deobfuscate("'a' + 'b';")
        self.assertIn("'ab'", result)

    def test_string_concat_nested(self):
        result = self._deobfuscate("'a' + 'b' + 'c';")
        self.assertIn("'abc'", result)

    def test_arithmetic_add(self):
        result = self._deobfuscate('2 + 3;')
        self.assertIn('5', result)

    def test_arithmetic_multiply(self):
        result = self._deobfuscate('10 * 2;')
        self.assertIn('20', result)

    def test_arithmetic_subtract(self):
        result = self._deobfuscate('10 - 3;')
        self.assertIn('7', result)

    def test_arithmetic_power(self):
        result = self._deobfuscate('2 ** 3;')
        self.assertIn('8', result)

    def test_arithmetic_modulo(self):
        result = self._deobfuscate('10 % 3;')
        self.assertIn('1', result)

    def test_arithmetic_bitwise_or(self):
        result = self._deobfuscate('5 | 3;')
        self.assertIn('7', result)

    def test_arithmetic_bitwise_and(self):
        result = self._deobfuscate('5 & 3;')
        self.assertIn('1', result)

    def test_arithmetic_bitwise_xor(self):
        result = self._deobfuscate('5 ^ 3;')
        self.assertIn('6', result)

    def test_arithmetic_left_shift(self):
        result = self._deobfuscate('1 << 3;')
        self.assertIn('8', result)

    def test_arithmetic_right_shift(self):
        result = self._deobfuscate('8 >> 2;')
        self.assertIn('2', result)

    def test_arithmetic_unsigned_right_shift(self):
        result = self._deobfuscate('(-1) >>> 0;')
        self.assertIn('4294967295', result)

    def test_arithmetic_division_by_zero_unchanged(self):
        result = self._deobfuscate('1 / 0;')
        self.assertIn('/', result)

    def test_tuple_all_literals(self):
        result = self._deobfuscate('("a", "b", "c");')
        self.assertIn('"c"', result)
        self.assertNotIn('"a"', result)

    def test_tuple_non_literal_unchanged(self):
        result = self._deobfuscate('("a", x, "c");')
        self.assertIn('x', result)

    def test_array_indexing(self):
        result = self._deobfuscate('["a", "b", "c"][1];')
        self.assertIn('"b"', result)

    def test_array_indexing_first(self):
        result = self._deobfuscate('["x", "y"][0];')
        self.assertIn('"x"', result)

    def test_bracket_to_dot(self):
        result = self._deobfuscate('obj["prop"];')
        self.assertIn('obj.prop', result)

    def test_bracket_non_identifier_unchanged(self):
        result = self._deobfuscate('obj["a-b"];')
        self.assertIn('"a-b"', result)

    def test_bracket_reserved_word_unchanged(self):
        result = self._deobfuscate('obj["class"];')
        self.assertIn('"class"', result)

    def test_paren_unwrap_string(self):
        result = self._deobfuscate('("hello");')
        self.assertIn('hello', result)
        self.assertNotIn('(', result.replace('"hello"', '').replace("'hello'", ''))

    def test_paren_unwrap_number(self):
        result = self._deobfuscate('(42);')
        self.assertIn('42', result)

    def test_unary_not_zero(self):
        result = self._deobfuscate('!0;')
        self.assertIn('true', result)

    def test_unary_not_one(self):
        result = self._deobfuscate('!1;')
        self.assertIn('false', result)

    def test_void_zero(self):
        result = self._deobfuscate('void 0;')
        self.assertIn('undefined', result)

    def test_typeof_string(self):
        result = self._deobfuscate('typeof "x";')
        self.assertIn("'string'", result)

    def test_typeof_number(self):
        result = self._deobfuscate('typeof 42;')
        self.assertIn("'number'", result)

    def test_typeof_boolean(self):
        result = self._deobfuscate('typeof true;')
        self.assertIn("'boolean'", result)

    def test_unary_negate(self):
        result = self._deobfuscate('-(5);')
        self.assertIn('-5', result)

    def test_unary_plus(self):
        result = self._deobfuscate('+(5);')
        self.assertIn('5', result)

    def test_non_constant_unchanged(self):
        result = self._deobfuscate('a + b;')
        self.assertIn('a + b', result)

    def test_non_constant_member_unchanged(self):
        result = self._deobfuscate('a[b];')
        self.assertIn('a[b]', result)

    def test_combined_deobfuscation(self):
        result = self._deobfuscate('var x = "hel" + "lo"; var y = [1, 2, 3][0];')
        self.assertIn("'hello'", result)
        self.assertIn('1', result)

    def test_make_string_literal_escapes_control_chars(self):
        node = make_string_literal('a\nb')
        self.assertEqual(node.raw, "'a\\nb'")
        node = make_string_literal('x\ry')
        self.assertEqual(node.raw, "'x\\ry'")
        node = make_string_literal('p\tq')
        self.assertEqual(node.raw, "'p\\tq'")
        node = make_string_literal('m\0n')
        self.assertEqual(node.raw, "'m\\0n'")

    def test_unescape_hex_space(self):
        result = self._deobfuscate("'hello\\x20world';")
        self.assertIn("'hello world'", result)

    def test_unescape_hex_mixed(self):
        result = self._deobfuscate("'A\\x42\\x0a\\x43';")
        self.assertIn('AB', result)
        self.assertIn('C', result)
        self.assertIn('\\x0a', result)

    def test_unescape_unicode_short(self):
        result = self._deobfuscate("'\\u0048\\u0069';")
        self.assertIn("'Hi'", result)

    def test_unescape_preserves_quote(self):
        result = self._deobfuscate("'don\\x27t';")
        self.assertIn('\\x27', result)

    def test_unescape_preserves_backslash(self):
        result = self._deobfuscate("'back\\x5cslash';")
        self.assertIn('\\x5c', result)


class TestStringArray(TestJsDeobfuscator):

    def test_string_array_default_preset(self):
        result = self._deobfuscate(
            r"var _0xe6abe5=_0x1b07;(function(_0x13a108,_0x20b5f6){var _0x2bca43=_0x1b07,_0x36965a=_0x13a108();whi"
            r"le(!![]){try{var _0x293699=-parseInt(_0x2bca43(0xa7))/0x1+-parseInt(_0x2bca43(0xa1))/0x2*(-parseInt("
            r"_0x2bca43(0xab))/0x3)+parseInt(_0x2bca43(0xa3))/0x4*(-parseInt(_0x2bca43(0xa9))/0x5)+parseInt(_0x2bc"
            r"a43(0xa6))/0x6+parseInt(_0x2bca43(0xaa))/0x7*(parseInt(_0x2bca43(0xa2))/0x8)+-parseInt(_0x2bca43(0xa"
            r"4))/0x9*(-parseInt(_0x2bca43(0xa5))/0xa)+-parseInt(_0x2bca43(0xa0))/0xb;if(_0x293699===_0x20b5f6)bre"
            r"ak;else _0x36965a['push'](_0x36965a['shift']());}catch(_0x35acf4){_0x36965a['push'](_0x36965a['shift"
            r"']());}}}(_0x2fc0,0x827c2));function _0x1b07(_0x3a2c1f,_0x271b5b){_0x3a2c1f=_0x3a2c1f-0xa0;var _0x2f"
            r"c00e=_0x2fc0();var _0x1b0775=_0x2fc00e[_0x3a2c1f];return _0x1b0775;}var msg=_0xe6abe5(0xac);function"
            r" _0x2fc0(){var _0x581e61=['2435007zbgngY','test\x20string','12767458FlCTYp','2BveYOA','96VHQLDe','16"
            r"0CSMRCB','486kcIkKD','183450npXmbZ','4067550xFhrYl','462884STmCds','log','50725EqKMLb','48769HzjsUR'"
            r"];_0x2fc0=function(){return _0x581e61;};return _0x2fc0();}console[_0xe6abe5(0xa8)](msg);"
        )
        self.assertIn("'test string'", result)
        self.assertIn('console.log', result)
        self.assertNotIn('_0x2fc0', result)

    def test_string_array_base64_encoding(self):
        source = (
            r"var _0x5dcc90=_0xf1a4;function _0xf1a4(_0x5939ce,_0x48b8f0){_0x5939ce=_0x5939ce-0x117;var _0x5943fc="
            r"_0x35ba();var _0x26baba=_0x5943fc[_0x5939ce];if(_0xf1a4['lvNcRA']===undefined){var _0x57faca=functio"
            r"n(_0x1e0002){var _0x35ba5c='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=';var _"
            r"0xf1a4db='',_0x147cac='',_0x1149e8=_0xf1a4db+_0x57faca,_0x574713=(''+function(){return 0x0;})['index"
            r"Of']('\x0a')!==-0x1;for(var _0x200f8a=0x0,_0x5b41ff,_0xeaa25,_0x4de86c=0x0;_0xeaa25=_0x1e0002['charA"
            r"t'](_0x4de86c++);~_0xeaa25&&(_0x5b41ff=_0x200f8a%0x4?_0x5b41ff*0x40+_0xeaa25:_0xeaa25,_0x200f8a++%0x"
            r"4)?_0xf1a4db+=_0x574713||_0x1149e8['charCodeAt'](_0x4de86c+0xa)-0xa!==0x0?String['fromCharCode'](0xf"
            r"f&_0x5b41ff>>(-0x2*_0x200f8a&0x6)):_0x200f8a:0x0){_0xeaa25=_0x35ba5c['indexOf'](_0xeaa25);}for(var _"
            r"0x2db414=0x0,_0x3d121a=_0xf1a4db['length'];_0x2db414<_0x3d121a;_0x2db414++){_0x147cac+='%'+('00'+_0x"
            r"f1a4db['charCodeAt'](_0x2db414)['toString'](0x10))['slice'](-0x2);}return decodeURIComponent(_0x147c"
            r"ac);};_0xf1a4['xXLpNw']=_0x57faca,_0xf1a4['odAnNy']={},_0xf1a4['lvNcRA']=!![];}var _0x543a02=_0x5943"
            r"fc[0x0],_0x588b78=_0x5939ce+_0x543a02,_0x69e6c5=_0xf1a4['odAnNy'][_0x588b78];if(!_0x69e6c5){var _0x3"
            r"8a721=function(_0x3e510d){this['KUwhtg']=_0x3e510d,this['lXYGTl']=[0x1,0x0,0x0],this['cSqHjm']=funct"
            r"ion(){return'newState';},this['clIxhe']='\x5cw+\x20*\x5c(\x5c)\x20*{\x5cw+\x20*',this['RNuSRa']='[\x"
            r"27|\x22].+[\x27|\x22];?\x20*}';};_0x38a721['prototype']['nyXfSO']=function(){var _0x11dd31=new RegEx"
            r"p(this['clIxhe']+this['RNuSRa']),_0x195f18=_0x11dd31['test'](this['cSqHjm']['toString']())?--this['l"
            r"XYGTl'][0x1]:--this['lXYGTl'][0x0];return this['LKokYd'](_0x195f18);},_0x38a721['prototype']['LKokYd"
            r"']=function(_0xf1d68b){if(!Boolean(~_0xf1d68b))return _0xf1d68b;return this['ARYADA'](this['KUwhtg']"
            r");},_0x38a721['prototype']['ARYADA']=function(_0x465b39){for(var _0x8dc7c8=0x0,_0x580cfb=this['lXYGT"
            r"l']['length'];_0x8dc7c8<_0x580cfb;_0x8dc7c8++){this['lXYGTl']['push'](Math['round'](Math['random']()"
            r")),_0x580cfb=this['lXYGTl']['length'];}return _0x465b39(this['lXYGTl'][0x0]);},(''+function(){return"
            r" 0x0;})['indexOf']('\x0a')===-0x1&&new _0x38a721(_0xf1a4)['nyXfSO'](),_0x26baba=_0xf1a4['xXLpNw'](_0"
            r"x26baba),_0xf1a4['odAnNy'][_0x588b78]=_0x26baba;}else _0x26baba=_0x69e6c5;return _0x26baba;}(functio"
            r"n(_0x521ca4,_0x3f2b97){var _0x201ef6=_0xf1a4,_0x139eba=_0x521ca4();while(!![]){try{var _0x51150c=-pa"
            r"rseInt(_0x201ef6(0x11a))/0x1*(parseInt(_0x201ef6(0x133))/0x2)+-parseInt(_0x201ef6(0x118))/0x3+parseI"
            r"nt(_0x201ef6(0x12d))/0x4+-parseInt(_0x201ef6(0x11f))/0x5*(-parseInt(_0x201ef6(0x12c))/0x6)+-parseInt"
            r"(_0x201ef6(0x117))/0x7*(parseInt(_0x201ef6(0x127))/0x8)+parseInt(_0x201ef6(0x119))/0x9*(parseInt(_0x"
            r"201ef6(0x128))/0xa)+parseInt(_0x201ef6(0x122))/0xb*(parseInt(_0x201ef6(0x12b))/0xc);if(_0x51150c===_"
            r"0x3f2b97)break;else _0x139eba['push'](_0x139eba['shift']());}catch(_0x250985){_0x139eba['push'](_0x1"
            r"39eba['shift']());}}}(_0x35ba,0x35dec));var _0x1e0002=(function(){var _0x4de86c=!![];return function"
            r"(_0x2db414,_0x3d121a){var _0x38a721=_0x4de86c?function(){if(_0x3d121a){var _0x3e510d=_0x3d121a['appl"
            r"y'](_0x2db414,arguments);return _0x3d121a=null,_0x3e510d;}}:function(){};return _0x4de86c=![],_0x38a"
            r"721;};}()),_0x69e6c5=_0x1e0002(this,function(){var _0x67f34f=_0xf1a4;return _0x69e6c5[_0x67f34f(0x11"
            r"c)]()['search'](_0x67f34f(0x11b))['toString']()[_0x67f34f(0x129)](_0x69e6c5)[_0x67f34f(0x11d)](_0x67"
            r"f34f(0x11b));});_0x69e6c5();var _0x57faca=(function(){var _0x11dd31=!![];return function(_0x195f18,_"
            r"0xf1d68b){var _0x465b39=_0x11dd31?function(){if(_0xf1d68b){var _0x8dc7c8=_0xf1d68b['apply'](_0x195f1"
            r"8,arguments);return _0xf1d68b=null,_0x8dc7c8;}}:function(){};return _0x11dd31=![],_0x465b39;};}()),_"
            r"0x26baba=_0x57faca(this,function(){var _0x7452c2=_0xf1a4,_0x580cfb=function(){var _0x4494ae=_0xf1a4,"
            r"_0xc3a305;try{_0xc3a305=Function(_0x4494ae(0x125)+_0x4494ae(0x123)+');')();}catch(_0x4bae61){_0xc3a3"
            r"05=window;}return _0xc3a305;},_0x18b0aa=_0x580cfb(),_0x29e29f=_0x18b0aa[_0x7452c2(0x12e)]=_0x18b0aa["
            r"_0x7452c2(0x12e)]||{},_0x2fdfc1=[_0x7452c2(0x11e),_0x7452c2(0x12a),'info',_0x7452c2(0x126),_0x7452c2"
            r"(0x132),_0x7452c2(0x12f),_0x7452c2(0x131)];for(var _0x331e55=0x0;_0x331e55<_0x2fdfc1[_0x7452c2(0x130"
            r")];_0x331e55++){var _0x3139c7=_0x57faca[_0x7452c2(0x129)][_0x7452c2(0x120)]['bind'](_0x57faca),_0x51"
            r"c297=_0x2fdfc1[_0x331e55],_0x49f6d5=_0x29e29f[_0x51c297]||_0x3139c7;_0x3139c7['__proto__']=_0x57faca"
            r"[_0x7452c2(0x124)](_0x57faca),_0x3139c7['toString']=_0x49f6d5[_0x7452c2(0x11c)]['bind'](_0x49f6d5),_"
            r"0x29e29f[_0x51c297]=_0x3139c7;}});_0x26baba();var msg=_0x5dcc90(0x121);function _0x35ba(){var _0x1da"
            r"a50=['Bg9N','mZvxufbrtwS','ChjVDg90ExbL','DgvZDcbZDhjPBMC','mte4nZeZmxDHz2HzEa','E30Uy29UC3rYDwn0B3i"
            r"OiNjLDhvYBIb0AgLZiIKOicK','yMLUza','CMv0DxjUicHMDw5JDgLVBIGPia','zxjYB3i','mJKZnNvdtKTLAa','mJy4ntu5"
            r"me5MzNzKyq','y29UC3rYDwn0B3i','D2fYBG','mtjzsfDvveO','mta3nJi4yMTssfvW','mtuXmdC3nKLPufvltG','y29UC2"
            r"9Szq','DgfIBgu','BgvUz3rO','DhjHy2u','zxHJzxb0Aw9U','mta1mtK4D1Ddq3nh','ndC0nM1lthvswq','mta3mJK4oxH"
            r"0BLbisW','owTJBujcwG','mwvNt0XVtW','kcGOlISPkYKRksSK','Dg9tDhjPBMC','C2vHCMnO'];_0x35ba=function(){r"
            r"eturn _0x1daa50;};return _0x35ba();}console['log'](msg);"
        )
        result = self._deobfuscate(source)
        self.assertIn("'test string'", result)
        self.assertIn('console.log', result)
        self.assertNotIn('_0x35ba', result)

    def test_string_array_rc4_encoding(self):
        source = (
            r"var _0x28eff0=_0x85f7;function _0x138c(){var _0x3144dc=['W4CnWRpcQKn3W7mbW4OU','W67dOZ3dU0hdS8ktzmom"
            r"','w24prCo1WPFdJCosWQ1zWQy','W7FdSCo6W5NdJa','W63dPZZcRrtcV8o+umo2g8krW6BcTW','WRxdHSooB8oIaspcNelcV"
            r"8oo','WP1+W7j1FXFdRNzpW6q/wG/cNCkmtSovrmoexrVdUSkYghRcLvmrW7LflCkw','ySkEW7S','W77cN8oI','jK9/baPXgt"
            r"FcGatcQGpcSG','WQdcUCo+W4tdL1tdJSoh','WPKtWRH4W4WAW6ddQbX1cWVcSa','W4vkpMHqW6dcNSk+W7qgW7Lwl8oYW5fjl"
            r"SkoWRm','vCkGfmoJFmoN','c8khWOJdKqbipgldPSooWOBcQa','pvVdR8kZWPnHuG','wmodW4RcI0O','WO5cBtWeW7frj8oR"
            r"','W6OGbG1sBmolA8ogyxLBuG','mmowW4reW4FdLSoDWR1bdSoPW6u','oSkmWO8X','W4yfWRRcPJ5LW6STW4OHW6O','W4pcU"
            r"KxdH8oC','W4njWPrtrXBcVeVcMWVdUa','ncPvfKyVsSkG','W4ygrSo5avpdKcS','dv1rdWa/WOv8vHi','hMPWqsrtW7agaq"
            r"','cmoKrCkk','WR/dHCk0dSkKE8kLBCkawc0','WR7cQCkIWPVcNH7cHSkJW6OKW593','W5jDohjq','WRDXwLmtna','W4ZcQ"
            r"moQW6tdUa','W6pdICo9W4NcRhmoFSkw','cmkdWO3dKajmnM7dJ8o5WRBcSW','WQddHCk2W6e+WP7dQCoyWPT8C8kzWRy','Cx"
            r"n+n0qrBmkb','WOapWQjVW5idW77cSK8Mzh4','W4pcThtdN8olzuag','W5XxWR8eW5BcUgmICfFcLSkD','v1/cSuxdHJRcQa'"
            r",'WPevW4mqAYlcSe/cLZe','p8ohW6tcO8kDWPhdPmol','WPxdHN/dOW'];_0x138c=function(){return _0x3144dc;};re"
            r"turn _0x138c();}function _0x85f7(_0x4af7e9,_0x789356){_0x4af7e9=_0x4af7e9-0x14c;var _0x7a46f0=_0x138"
            r"c();var _0x5eebe1=_0x7a46f0[_0x4af7e9];if(_0x85f7['Ecnfls']===undefined){var _0x5083fb=function(_0x1"
            r"38cc6){var _0x85f78b='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=';var _0x4396"
            r"3a='',_0x3e8610='',_0x3e54d4=_0x43963a+_0x5083fb,_0x29f804=(''+function(){return 0x0;})['indexOf']('"
            r"\x0a')!==-0x1;for(var _0x518262=0x0,_0x50edcc,_0x1a6a9b,_0x1b65be=0x0;_0x1a6a9b=_0x138cc6['charAt']("
            r"_0x1b65be++);~_0x1a6a9b&&(_0x50edcc=_0x518262%0x4?_0x50edcc*0x40+_0x1a6a9b:_0x1a6a9b,_0x518262++%0x4"
            r")?_0x43963a+=_0x29f804||_0x3e54d4['charCodeAt'](_0x1b65be+0xa)-0xa!==0x0?String['fromCharCode'](0xff"
            r"&_0x50edcc>>(-0x2*_0x518262&0x6)):_0x518262:0x0){_0x1a6a9b=_0x85f78b['indexOf'](_0x1a6a9b);}for(var "
            r"_0x21a7b2=0x0,_0x2450dc=_0x43963a['length'];_0x21a7b2<_0x2450dc;_0x21a7b2++){_0x3e8610+='%'+('00'+_0"
            r"x43963a['charCodeAt'](_0x21a7b2)['toString'](0x10))['slice'](-0x2);}return decodeURIComponent(_0x3e8"
            r"610);};var _0x42cc8d=function(_0x525d21,_0xb4ae49){var _0x5812fb=[],_0x42d0ca=0x0,_0x72b3ac,_0x7cffb"
            r"e='';_0x525d21=_0x5083fb(_0x525d21);var _0x347b99;for(_0x347b99=0x0;_0x347b99<0x100;_0x347b99++){_0x"
            r"5812fb[_0x347b99]=_0x347b99;}for(_0x347b99=0x0;_0x347b99<0x100;_0x347b99++){_0x42d0ca=(_0x42d0ca+_0x"
            r"5812fb[_0x347b99]+_0xb4ae49['charCodeAt'](_0x347b99%_0xb4ae49['length']))%0x100,_0x72b3ac=_0x5812fb["
            r"_0x347b99],_0x5812fb[_0x347b99]=_0x5812fb[_0x42d0ca],_0x5812fb[_0x42d0ca]=_0x72b3ac;}_0x347b99=0x0,_"
            r"0x42d0ca=0x0;for(var _0x55c939=0x0;_0x55c939<_0x525d21['length'];_0x55c939++){_0x347b99=(_0x347b99+0"
            r"x1)%0x100,_0x42d0ca=(_0x42d0ca+_0x5812fb[_0x347b99])%0x100,_0x72b3ac=_0x5812fb[_0x347b99],_0x5812fb["
            r"_0x347b99]=_0x5812fb[_0x42d0ca],_0x5812fb[_0x42d0ca]=_0x72b3ac,_0x7cffbe+=String['fromCharCode'](_0x"
            r"525d21['charCodeAt'](_0x55c939)^_0x5812fb[(_0x5812fb[_0x347b99]+_0x5812fb[_0x42d0ca])%0x100]);}retur"
            r"n _0x7cffbe;};_0x85f7['WMPOrt']=_0x42cc8d,_0x85f7['HNjmjk']={},_0x85f7['Ecnfls']=!![];}var _0x1897c4"
            r"=_0x7a46f0[0x0],_0x55963d=_0x4af7e9+_0x1897c4,_0x7c3c=_0x85f7['HNjmjk'][_0x55963d];if(!_0x7c3c){if(_"
            r"0x85f7['HYiIOj']===undefined){var _0x1c604c=function(_0x39dfaa){this['ekjwSF']=_0x39dfaa,this['thDCz"
            r"d']=[0x1,0x0,0x0],this['MrvPnY']=function(){return'newState';},this['iUjMCk']='\x5cw+\x20*\x5c(\x5c)"
            r"\x20*{\x5cw+\x20*',this['pKSLss']='[\x27|\x22].+[\x27|\x22];?\x20*}';};_0x1c604c['prototype']['rZEHh"
            r"c']=function(){var _0x24be40=new RegExp(this['iUjMCk']+this['pKSLss']),_0x508196=_0x24be40['test'](t"
            r"his['MrvPnY']['toString']())?--this['thDCzd'][0x1]:--this['thDCzd'][0x0];return this['reqeIL'](_0x50"
            r"8196);},_0x1c604c['prototype']['reqeIL']=function(_0x3cea1d){if(!Boolean(~_0x3cea1d))return _0x3cea1"
            r"d;return this['eMIAaD'](this['ekjwSF']);},_0x1c604c['prototype']['eMIAaD']=function(_0x5a4a1f){for(v"
            r"ar _0x404b8a=0x0,_0x2330df=this['thDCzd']['length'];_0x404b8a<_0x2330df;_0x404b8a++){this['thDCzd']["
            r"'push'](Math['round'](Math['random']())),_0x2330df=this['thDCzd']['length'];}return _0x5a4a1f(this['"
            r"thDCzd'][0x0]);},(''+function(){return 0x0;})['indexOf']('\x0a')===-0x1&&new _0x1c604c(_0x85f7)['rZE"
            r"Hhc'](),_0x85f7['HYiIOj']=!![];}_0x5eebe1=_0x85f7['WMPOrt'](_0x5eebe1,_0x789356),_0x85f7['HNjmjk'][_"
            r"0x55963d]=_0x5eebe1;}else _0x5eebe1=_0x7c3c;return _0x5eebe1;}(function(_0x566ec1,_0x397185){var _0x"
            r"1fe482=_0x85f7,_0x50fe64=_0x566ec1();while(!![]){try{var _0xafcaf6=parseInt(_0x1fe482(0x175,'43HA'))"
            r"/0x1*(-parseInt(_0x1fe482(0x150,'cGSY'))/0x2)+-parseInt(_0x1fe482(0x173,'DYhn'))/0x3*(parseInt(_0x1f"
            r"e482(0x15a,'vHRQ'))/0x4)+-parseInt(_0x1fe482(0x162,'cq*e'))/0x5+parseInt(_0x1fe482(0x159,'p]TC'))/0x"
            r"6+parseInt(_0x1fe482(0x174,'q1WL'))/0x7+-parseInt(_0x1fe482(0x161,'Dja*'))/0x8*(-parseInt(_0x1fe482("
            r"0x15e,'DYhn'))/0x9)+parseInt(_0x1fe482(0x14d,'zupi'))/0xa*(parseInt(_0x1fe482(0x16d,'Ot85'))/0xb);if"
            r"(_0xafcaf6===_0x397185)break;else _0x50fe64['push'](_0x50fe64['shift']());}catch(_0x1fe138){_0x50fe6"
            r"4['push'](_0x50fe64['shift']());}}}(_0x138c,0x697ae));var _0x42cc8d=(function(){var _0x1a6a9b=!![];r"
            r"eturn function(_0x1b65be,_0x21a7b2){var _0x2450dc=_0x1a6a9b?function(){var _0x570a5c=_0x85f7;if(_0x2"
            r"1a7b2){var _0x525d21=_0x21a7b2[_0x570a5c(0x160,'DYhn')](_0x1b65be,arguments);return _0x21a7b2=null,_"
            r"0x525d21;}}:function(){};return _0x1a6a9b=![],_0x2450dc;};}()),_0x7c3c=_0x42cc8d(this,function(){var"
            r" _0x36d5f=_0x85f7;return _0x7c3c['toString']()['search'](_0x36d5f(0x16e,'vHRQ'))[_0x36d5f(0x168,'43H"
            r"A')]()[_0x36d5f(0x155,'^WN@')](_0x7c3c)[_0x36d5f(0x15d,'dChY')](_0x36d5f(0x176,'1NBB'));});_0x7c3c()"
            r";var _0x5083fb=(function(){var _0xb4ae49=!![];return function(_0x5812fb,_0x42d0ca){var _0x72b3ac=_0x"
            r"b4ae49?function(){var _0x5001a2=_0x85f7;if(_0x42d0ca){var _0x7cffbe=_0x42d0ca[_0x5001a2(0x153,'vHRQ'"
            r")](_0x5812fb,arguments);return _0x42d0ca=null,_0x7cffbe;}}:function(){};return _0xb4ae49=![],_0x72b3"
            r"ac;};}()),_0x5eebe1=_0x5083fb(this,function(){var _0x4dc929=_0x85f7,_0x347b99;try{var _0x55c939=Func"
            r"tion(_0x4dc929(0x15c,']!wI')+_0x4dc929(0x156,'BqaA')+');');_0x347b99=_0x55c939();}catch(_0x404b8a){_"
            r"0x347b99=window;}var _0x1c604c=_0x347b99[_0x4dc929(0x15f,'FdCr')]=_0x347b99[_0x4dc929(0x14c,'RT!O')]"
            r"||{},_0x39dfaa=[_0x4dc929(0x157,'PGZ!'),_0x4dc929(0x14f,'ttcu'),_0x4dc929(0x16c,'4lCo'),_0x4dc929(0x"
            r"16f,']!wI'),_0x4dc929(0x16b,'5)wT'),_0x4dc929(0x166,'KtWU'),_0x4dc929(0x171,'CTYV')];for(var _0x24be"
            r"40=0x0;_0x24be40<_0x39dfaa[_0x4dc929(0x170,'cq*e')];_0x24be40++){var _0x508196=_0x5083fb[_0x4dc929(0"
            r"x167,'zupi')]['prototype']['bind'](_0x5083fb),_0x3cea1d=_0x39dfaa[_0x24be40],_0x5a4a1f=_0x1c604c[_0x"
            r"3cea1d]||_0x508196;_0x508196[_0x4dc929(0x172,'NehO')]=_0x5083fb[_0x4dc929(0x164,'KUmh')](_0x5083fb),"
            r"_0x508196['toString']=_0x5a4a1f[_0x4dc929(0x177,'KtWU')]['bind'](_0x5a4a1f),_0x1c604c[_0x3cea1d]=_0x"
            r"508196;}});_0x5eebe1();var msg=_0x28eff0(0x152,'#Y8%');console[_0x28eff0(0x158,'q1WL')](msg);"
        )
        result = self._deobfuscate(source)
        self.assertIn("'test string'", result)
        self.assertIn('console.log', result)
        self.assertNotIn('_0x138c', result)
