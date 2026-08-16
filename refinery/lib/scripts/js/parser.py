from __future__ import annotations

from contextlib import contextmanager

from refinery.lib.scripts.js.lexer import (
    JsLexer,
    JsLexerState,
    decode_js_string_body,
    decode_js_template_body,
)
from refinery.lib.scripts.js.model import (
    Expression,
    JsArrayExpression,
    JsArrayPattern,
    JsArrowFunctionExpression,
    JsAssignmentExpression,
    JsAssignmentPattern,
    JsAwaitExpression,
    JsBigIntLiteral,
    JsBinaryExpression,
    JsBlockStatement,
    JsBooleanLiteral,
    JsBreakStatement,
    JsCallExpression,
    JsCatchClause,
    JsClassBody,
    JsClassDeclaration,
    JsClassExpression,
    JsConditionalExpression,
    JsContinueStatement,
    JsDebuggerStatement,
    JsDecorator,
    JsDoWhileStatement,
    JsEmptyStatement,
    JsErrorNode,
    JsExportAllDeclaration,
    JsExportDefaultDeclaration,
    JsExportNamedDeclaration,
    JsExportSpecifier,
    JsExpressionStatement,
    JsForInStatement,
    JsForOfStatement,
    JsForStatement,
    JsFunctionDeclaration,
    JsFunctionExpression,
    JsIdentifier,
    JsIfStatement,
    JsImportAttribute,
    JsImportDeclaration,
    JsImportDefaultSpecifier,
    JsImportExpression,
    JsImportNamespaceSpecifier,
    JsImportSpecifier,
    JsLabeledStatement,
    JsLogicalExpression,
    JsMemberExpression,
    JsMetaProperty,
    JsMethodDefinition,
    JsMethodKind,
    JsNewExpression,
    JsNullLiteral,
    JsNumericLiteral,
    JsObjectExpression,
    JsObjectPattern,
    JsParenthesizedExpression,
    JsPrivateIdentifier,
    JsProperty,
    JsPropertyDefinition,
    JsPropertyKind,
    JsRegExpLiteral,
    JsRestElement,
    JsReturnStatement,
    JsScript,
    JsSequenceExpression,
    JsSpreadElement,
    JsStaticBlock,
    JsStringLiteral,
    JsSwitchCase,
    JsSwitchStatement,
    JsTaggedTemplateExpression,
    JsTemplateElement,
    JsTemplateLiteral,
    JsThisExpression,
    JsThrowStatement,
    JsTryStatement,
    JsUnaryExpression,
    JsUpdateExpression,
    JsVariableDeclaration,
    JsVariableDeclarator,
    JsVarKind,
    JsWhileStatement,
    JsWithStatement,
    JsYieldExpression,
    Statement,
)
from refinery.lib.scripts.js.strict import mark_directives
from refinery.lib.scripts.js.token import JsToken, JsTokenKind

_PREC_EXPONENTIATION = 15

_BINARY_PREC: dict[JsTokenKind, tuple[int, bool]] = {
    JsTokenKind.QQ:         ( 4, True),   # noqa
    JsTokenKind.OR:         ( 5, True),   # noqa
    JsTokenKind.AND:        ( 6, True),   # noqa
    JsTokenKind.PIPE:       ( 7, False),  # noqa
    JsTokenKind.CARET:      ( 8, False),  # noqa
    JsTokenKind.AMP:        ( 9, False),  # noqa
    JsTokenKind.EQ2:        (10, False),  # noqa
    JsTokenKind.BANG_EQ:    (10, False),  # noqa
    JsTokenKind.EQ3:        (10, False),  # noqa
    JsTokenKind.BANG_EQ2:   (10, False),  # noqa
    JsTokenKind.LT:         (11, False),  # noqa
    JsTokenKind.GT:         (11, False),  # noqa
    JsTokenKind.LT_EQ:      (11, False),  # noqa
    JsTokenKind.GT_EQ:      (11, False),  # noqa
    JsTokenKind.INSTANCEOF: (11, False),  # noqa
    JsTokenKind.IN:         (11, False),  # noqa
    JsTokenKind.LT2:        (12, False),  # noqa
    JsTokenKind.GT2:        (12, False),  # noqa
    JsTokenKind.GT3:        (12, False),  # noqa
    JsTokenKind.PLUS:       (13, False),  # noqa
    JsTokenKind.MINUS:      (13, False),  # noqa
    JsTokenKind.STAR:       (14, False),  # noqa
    JsTokenKind.SLASH:      (14, False),  # noqa
    JsTokenKind.PERCENT:    (14, False),  # noqa
    JsTokenKind.STAR2:      (_PREC_EXPONENTIATION, False), # noqa
}

_VAR_KIND_MAP: dict[JsTokenKind, JsVarKind] = {
    JsTokenKind.VAR:   JsVarKind.VAR,    # noqa
    JsTokenKind.LET:   JsVarKind.LET,    # noqa
    JsTokenKind.CONST: JsVarKind.CONST,  # noqa
}

_PROP_KIND_MAP: dict[str, JsPropertyKind] = {
    'get': JsPropertyKind.GET,
    'set': JsPropertyKind.SET,
}


class JsParser:

    @staticmethod
    def _parse_int_text(text: str) -> int:
        if text.startswith(('0x', '0X')):
            return int(text, 16)
        if text.startswith(('0o', '0O')):
            return int(text, 8)
        if text.startswith(('0b', '0B')):
            return int(text, 2)
        if len(text) > 1 and text[0] == '0' and all(d in '01234567' for d in text):
            return int(text, 8)
        return int(text)

    def __init__(self, source: str, *, top_level_await: bool = False):
        self._lexer = JsLexer(source)
        self._source = source
        self._tokens = self._lexer.tokenize()
        self._current: JsToken = JsToken(JsTokenKind.EOF, '', 0)
        self._preceded_by_newline: bool = False
        self._ahead: JsToken | None = None
        self._ahead_newline: bool = False
        self._ahead_state: tuple[JsLexerState, int] | None = None
        self._no_in: bool = False
        self._in_async: bool = top_level_await
        self._in_generator: bool = False
        self._pending_comments: list[str] = []
        self._advance()

    def _pull_token(self) -> tuple[JsToken, bool]:
        had_newline = False
        while True:
            tok = next(self._tokens, JsToken(JsTokenKind.EOF, '', len(self._source)))
            if tok.kind == JsTokenKind.NEWLINE:
                had_newline = True
                continue
            if tok.kind == JsTokenKind.COMMENT:
                self._pending_comments.append(tok.value)
                continue
            break
        return tok, had_newline

    def _advance(self) -> JsToken:
        prev = self._current
        if self._ahead is not None:
            self._current = self._ahead
            self._preceded_by_newline = self._ahead_newline
            self._ahead = None
            self._ahead_state = None
            return prev
        self._current, self._preceded_by_newline = self._pull_token()
        return prev

    def _drain_comments(self, node):
        if self._pending_comments:
            node.leading_comments.extend(self._pending_comments)
            self._pending_comments.clear()

    def _peek(self) -> JsToken:
        return self._current

    def _peek_next(self) -> JsToken:
        if self._ahead is None:
            self._ahead_state = self._lexer.capture(), len(self._pending_comments)
            self._ahead, self._ahead_newline = self._pull_token()
        return self._ahead

    def _rescan_as_regexp(self) -> JsToken:
        """
        Read the slash the parser is holding again, as the regular expression it begins. A slash is
        the one character whose token depends on where the grammar stands rather than on what the
        text says, and the lexer is not standing anywhere: it spells every slash as an operator, and
        this is the single place that knows that an expression is about to begin. Rewinding is what
        makes that affordable — reading a division that was a regular expression costs one token,
        whereas the other way around has already swallowed the rest of the line.

        Everything scanned since the slash is given back, because a lookahead token may have opened
        or closed a template hole and a skipped comment would otherwise be collected twice. The
        slash itself is given back too where no literal begins there, so that a scan which found no
        terminator on its line leaves the operator standing rather than a literal nobody wrote.
        """
        if self._ahead_state is None:
            state, comments = self._lexer.capture(), len(self._pending_comments)
        else:
            state, comments = self._ahead_state
        resume_pos, resume_state = self._lexer.pos, self._lexer.capture()
        self._lexer.rewind(self._current.offset, state)
        token = self._lexer.scan_regexp()
        if token is None:
            self._lexer.rewind(resume_pos, resume_state)
            return self._current
        del self._pending_comments[comments:]
        self._ahead = None
        self._ahead_newline = False
        self._ahead_state = None
        self._current = token
        self._tokens = self._lexer.tokenize()
        return self._current

    def _at(self, *kinds: JsTokenKind) -> bool:
        return self._current.kind in kinds

    def _eat(self, kind: JsTokenKind) -> JsToken | None:
        if self._current.kind == kind:
            return self._advance()
        return None

    def _expect(self, kind: JsTokenKind) -> JsToken:
        if self._current.kind == kind:
            return self._advance()
        tok = self._current
        self._advance()
        return JsToken(kind, tok.value, tok.offset, tok.terminated)

    def _is_binding_identifier(self, token: JsToken) -> bool:
        """
        Whether the token can serve as an ordinary binding or reference name. Several contextual
        keywords (`as`, `from`, `of`, `let`, `async`) are always valid names, while `await` and
        `yield` are valid names except inside an async function or a generator respectively. This is
        the identifier acceptance of `_parse_primary_expression` itself, so a name-reading site
        accepts exactly the tokens the expression grammar would treat as a reference.
        """
        kind = token.kind
        return (
            kind in (
                JsTokenKind.IDENTIFIER,
                JsTokenKind.AS,
                JsTokenKind.FROM,
                JsTokenKind.OF,
                JsTokenKind.LET,
                JsTokenKind.ASYNC,
            )
            or (kind is JsTokenKind.AWAIT and not self._in_async)
            or (kind is JsTokenKind.YIELD and not self._in_generator)
        )

    def _at_binding_identifier(self) -> bool:
        return self._is_binding_identifier(self._current)

    def _at_variable_declaration(self) -> bool:
        """
        Whether a variable declaration begins here, rather than an expression that merely opens with
        the same word. ECMA-262 reserves `let` in strict code only, so wherever a statement may also
        be read as an expression, a `let` declares nothing unless a binding follows it: it is a name
        being called in `let(1)`, divided in `let / 2` and read in `let.a`, and only `let [` is the
        spelling a statement is forbidden to take as an expression.
        """
        if self._at(JsTokenKind.VAR, JsTokenKind.CONST):
            return True
        if not self._at(JsTokenKind.LET):
            return False
        ahead = self._peek_next()
        return (
            self._is_binding_identifier(ahead)
            or ahead.kind in (JsTokenKind.LBRACKET, JsTokenKind.LBRACE)
        )

    def _eat_semicolon(self) -> bool:
        if self._eat(JsTokenKind.SEMICOLON):
            return True
        if self._at(JsTokenKind.RBRACE, JsTokenKind.EOF):
            return True
        if self._preceded_by_newline:
            return True
        return False

    @contextmanager
    def _with_no_in(self, value: bool):
        saved = self._no_in
        self._no_in = value
        try:
            yield
        finally:
            self._no_in = saved

    @contextmanager
    def _function_body_context(self, is_async: bool, is_generator: bool):
        saved = (self._in_async, self._in_generator)
        self._in_async = is_async
        self._in_generator = is_generator
        try:
            yield
        finally:
            self._in_async, self._in_generator = saved

    def parse(self) -> JsScript:
        script = self._parse_program()
        mark_directives(script)
        return script

    def _parse_statement_list(self, *stop: JsTokenKind) -> list[Statement]:
        body: list[Statement] = []
        while not self._at(*stop):
            mark = self._current.offset
            comments = list(self._pending_comments)
            self._pending_comments.clear()
            try:
                stmt = self._parse_statement()
            except Exception:
                stmt = None
            if stmt is not None:
                stmt.leading_comments.extend(comments)
                body.append(stmt)
            elif self._current.offset == mark:
                tok = self._advance()
                error = JsErrorNode(offset=tok.offset, text=tok.value)
                error.leading_comments.extend(comments)
                body.append(error)
        return body

    def _parse_program(self) -> JsScript:
        offset = self._current.offset
        body = self._parse_statement_list(JsTokenKind.EOF)
        return JsScript(body=body, offset=offset)

    def _parse_statement(self) -> Statement | None:
        offset = self._current.offset
        kind = self._current.kind

        if kind == JsTokenKind.LBRACE:
            return self._parse_block_statement()
        if kind == JsTokenKind.SEMICOLON:
            self._advance()
            return JsEmptyStatement(offset=offset)
        if self._at_variable_declaration():
            return self._parse_variable_declaration()
        if kind == JsTokenKind.IF:
            return self._parse_if_statement()
        if kind == JsTokenKind.WHILE:
            return self._parse_while_statement()
        if kind == JsTokenKind.DO:
            return self._parse_do_while_statement()
        if kind == JsTokenKind.FOR:
            return self._parse_for_statement()
        if kind == JsTokenKind.SWITCH:
            return self._parse_switch_statement()
        if kind == JsTokenKind.TRY:
            return self._parse_try_statement()
        if kind == JsTokenKind.WITH:
            return self._parse_with_statement()
        if kind == JsTokenKind.RETURN:
            return self._parse_return_statement()
        if kind == JsTokenKind.THROW:
            return self._parse_throw_statement()
        if kind == JsTokenKind.BREAK:
            return self._parse_break_statement()
        if kind == JsTokenKind.CONTINUE:
            return self._parse_continue_statement()
        if kind == JsTokenKind.FUNCTION:
            return self._parse_function_declaration()
        if kind == JsTokenKind.AT:
            decorators = self._parse_decorators()
            if self._at(JsTokenKind.EXPORT):
                return self._parse_export_declaration(decorators)
            if self._at(JsTokenKind.CLASS):
                return self._parse_class_declaration(decorators)
            return JsErrorNode(
                text=self._source[offset:self._current.offset].rstrip(),
                message='decorators must precede a class',
                offset=offset,
            )
        if kind == JsTokenKind.CLASS:
            return self._parse_class_declaration()
        if kind == JsTokenKind.DEBUGGER:
            self._advance()
            self._eat_semicolon()
            return JsDebuggerStatement(offset=offset)
        if kind == JsTokenKind.IMPORT and self._peek_next().kind not in (
            JsTokenKind.LPAREN, JsTokenKind.DOT,
        ):
            return self._parse_import_declaration()
        if kind == JsTokenKind.EXPORT:
            return self._parse_export_declaration()
        if self._at_async_function():
            self._advance()
            return self._parse_function_declaration(is_async=True)

        expr = self._parse_expression()

        if (
            isinstance(expr, JsIdentifier)
            and self._eat(JsTokenKind.COLON)
        ):
            body = self._parse_statement()
            return JsLabeledStatement(label=expr, body=body, offset=offset)

        self._eat_semicolon()
        if isinstance(expr, JsErrorNode):
            return expr
        return JsExpressionStatement(expression=expr, offset=offset)

    def _parse_block_statement(self) -> JsBlockStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.LBRACE)
        body = self._parse_statement_list(JsTokenKind.RBRACE, JsTokenKind.EOF)
        self._expect(JsTokenKind.RBRACE)
        return JsBlockStatement(body=body, offset=offset)

    def _parse_variable_declaration(self) -> JsVariableDeclaration:
        offset = self._current.offset
        kind_tok = self._advance()
        kind = _VAR_KIND_MAP[kind_tok.kind]
        declarations: list[JsVariableDeclarator] = []
        declarations.append(self._parse_variable_declarator())
        while self._eat(JsTokenKind.COMMA):
            declarations.append(self._parse_variable_declarator())
        self._eat_semicolon()
        return JsVariableDeclaration(declarations=declarations, kind=kind, offset=offset)

    def _parse_variable_declarator(self) -> JsVariableDeclarator:
        offset = self._current.offset
        id_node = self._parse_binding_pattern()
        init = None
        if self._eat(JsTokenKind.EQUALS):
            init = self._parse_assignment_expression()
        return JsVariableDeclarator(id=id_node, init=init, offset=offset)

    def _parse_binding_pattern(self) -> Expression:
        if self._at(JsTokenKind.LBRACKET):
            return self._parse_array_pattern()
        if self._at(JsTokenKind.LBRACE):
            return self._parse_object_pattern()
        return self._parse_binding_identifier()

    def _parse_binding_identifier(self) -> Expression:
        offset = self._current.offset
        if self._at_binding_identifier():
            tok = self._advance()
        else:
            tok = self._expect(JsTokenKind.IDENTIFIER)
        return self._name_or_error(tok.value, offset)

    def _name_or_error(self, name: str, offset: int) -> Expression:
        """
        The name a token spells, where it spells one. A file that ends in the middle of a
        declaration leaves the position a name was expected in holding nothing, and a name spelled
        by nothing has no text at all: printing it closes the source up over the gap, so `var` at
        the end of a file would come back as `var ;`. What is handed back instead is the span
        itself, which prints as what was written and states that the parser did not read it.
        """
        if name:
            return JsIdentifier(name=name, offset=offset)
        return JsErrorNode(text=name, message='expected a name', offset=offset)

    def _parse_array_pattern(self) -> JsArrayPattern:
        offset = self._current.offset
        self._expect(JsTokenKind.LBRACKET)
        elements: list[Expression | None] = []
        while not self._at(JsTokenKind.RBRACKET, JsTokenKind.EOF):
            if self._at(JsTokenKind.COMMA):
                elements.append(None)
                self._advance()
                continue
            if self._at(JsTokenKind.ELLIPSIS):
                elements.append(self._parse_rest_element())
                break
            elem = self._parse_binding_pattern()
            if self._eat(JsTokenKind.EQUALS):
                right = self._parse_assignment_expression()
                elem = JsAssignmentPattern(left=elem, right=right, offset=elem.offset)
            elements.append(elem)
            if not self._at(JsTokenKind.RBRACKET):
                self._expect(JsTokenKind.COMMA)
        self._expect(JsTokenKind.RBRACKET)
        return JsArrayPattern(elements=elements, offset=offset)

    def _parse_object_pattern(self) -> JsObjectPattern:
        offset = self._current.offset
        self._expect(JsTokenKind.LBRACE)
        properties: list[JsProperty | JsRestElement] = []
        while not self._at(JsTokenKind.RBRACE, JsTokenKind.EOF):
            if self._at(JsTokenKind.ELLIPSIS):
                properties.append(self._parse_rest_element())
                break
            prop = self._parse_object_pattern_property()
            properties.append(prop)
            if not self._at(JsTokenKind.RBRACE):
                self._expect(JsTokenKind.COMMA)
        self._expect(JsTokenKind.RBRACE)
        return JsObjectPattern(properties=properties, offset=offset)

    def _parse_object_pattern_property(self) -> JsProperty:
        offset = self._current.offset
        if self._at(JsTokenKind.LBRACKET):
            self._advance()
            key = self._parse_assignment_expression()
            self._expect(JsTokenKind.RBRACKET)
            self._expect(JsTokenKind.COLON)
            value = self._parse_binding_pattern()
            if self._eat(JsTokenKind.EQUALS):
                right = self._parse_assignment_expression()
                value = JsAssignmentPattern(left=value, right=right, offset=value.offset)
            return JsProperty(
                key=key, value=value, computed=True, shorthand=False, offset=offset)

        key = self._parse_property_name()
        if self._eat(JsTokenKind.COLON):
            value = self._parse_binding_pattern()
            if self._eat(JsTokenKind.EQUALS):
                right = self._parse_assignment_expression()
                value = JsAssignmentPattern(left=value, right=right, offset=value.offset)
            return JsProperty(
                key=key, value=value, computed=False, shorthand=False, offset=offset)

        value = key
        if self._eat(JsTokenKind.EQUALS):
            right = self._parse_assignment_expression()
            value = JsAssignmentPattern(left=key, right=right, offset=key.offset)
        return JsProperty(key=key, value=value, computed=False, shorthand=True, offset=offset)

    def _parse_rest_element(self) -> JsRestElement:
        offset = self._current.offset
        self._expect(JsTokenKind.ELLIPSIS)
        argument = self._parse_binding_pattern()
        return JsRestElement(argument=argument, offset=offset)

    def _parse_if_statement(self) -> JsIfStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.IF)
        self._expect(JsTokenKind.LPAREN)
        test = self._parse_expression()
        self._expect(JsTokenKind.RPAREN)
        consequent = self._parse_statement()
        alternate = None
        if self._eat(JsTokenKind.ELSE):
            alternate = self._parse_statement()
        return JsIfStatement(
            test=test, consequent=consequent, alternate=alternate, offset=offset)

    def _parse_while_statement(self) -> JsWhileStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.WHILE)
        self._expect(JsTokenKind.LPAREN)
        test = self._parse_expression()
        self._expect(JsTokenKind.RPAREN)
        body = self._parse_statement()
        return JsWhileStatement(test=test, body=body, offset=offset)

    def _parse_do_while_statement(self) -> JsDoWhileStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.DO)
        body = self._parse_statement()
        self._expect(JsTokenKind.WHILE)
        self._expect(JsTokenKind.LPAREN)
        test = self._parse_expression()
        self._expect(JsTokenKind.RPAREN)
        self._eat_semicolon()
        return JsDoWhileStatement(test=test, body=body, offset=offset)

    def _parse_for_statement(self) -> Statement:
        offset = self._current.offset
        self._expect(JsTokenKind.FOR)

        is_await = False
        if self._eat(JsTokenKind.AWAIT):
            is_await = True

        self._expect(JsTokenKind.LPAREN)

        if self._at(JsTokenKind.SEMICOLON):
            self._advance()
            return self._parse_for_rest(None, offset)

        if self._at_variable_declaration():
            decl_offset = self._current.offset
            kind_tok = self._advance()
            kind = _VAR_KIND_MAP[kind_tok.kind]
            with self._with_no_in(True):
                declarator = self._parse_variable_declarator()
            decl = JsVariableDeclaration(
                declarations=[declarator], kind=kind, offset=decl_offset)
            result = self._parse_for_in_or_of(decl, is_await, offset)
            if result is not None:
                return result
            while self._eat(JsTokenKind.COMMA):
                with self._with_no_in(True):
                    decl.declarations.append(self._parse_variable_declarator())
            self._expect(JsTokenKind.SEMICOLON)
            return self._parse_for_rest(decl, offset)

        with self._with_no_in(True):
            init_expr = self._parse_expression()
        result = self._parse_for_in_or_of(init_expr, is_await, offset)
        if result is not None:
            return result
        self._expect(JsTokenKind.SEMICOLON)
        return self._parse_for_rest(init_expr, offset)

    def _parse_for_in_or_of(
        self,
        left: Expression | Statement,
        is_await: bool,
        offset: int,
    ) -> JsForInStatement | JsForOfStatement | None:
        if self._eat(JsTokenKind.IN):
            right = self._parse_expression()
            self._expect(JsTokenKind.RPAREN)
            body = self._parse_statement()
            return JsForInStatement(left=left, right=right, body=body, offset=offset)
        if self._at(JsTokenKind.OF):
            self._advance()
            right = self._parse_assignment_expression()
            self._expect(JsTokenKind.RPAREN)
            body = self._parse_statement()
            return JsForOfStatement(
                left=left, right=right, body=body, is_await=is_await, offset=offset)
        return None

    def _parse_for_rest(
        self,
        init: Expression | Statement | None,
        offset: int,
    ) -> JsForStatement:
        test = None
        if not self._at(JsTokenKind.SEMICOLON):
            test = self._parse_expression()
        self._expect(JsTokenKind.SEMICOLON)
        update = None
        if not self._at(JsTokenKind.RPAREN):
            update = self._parse_expression()
        self._expect(JsTokenKind.RPAREN)
        body = self._parse_statement()
        return JsForStatement(
            init=init, test=test, update=update, body=body, offset=offset)

    def _parse_switch_statement(self) -> JsSwitchStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.SWITCH)
        self._expect(JsTokenKind.LPAREN)
        discriminant = self._parse_expression()
        self._expect(JsTokenKind.RPAREN)
        self._expect(JsTokenKind.LBRACE)
        cases: list[JsSwitchCase] = []
        while not self._at(JsTokenKind.RBRACE, JsTokenKind.EOF):
            cases.append(self._parse_switch_case())
        self._expect(JsTokenKind.RBRACE)
        return JsSwitchStatement(
            discriminant=discriminant, cases=cases, offset=offset)

    def _parse_switch_case(self) -> JsSwitchCase:
        offset = self._current.offset
        test = None
        if self._eat(JsTokenKind.CASE):
            test = self._parse_expression()
            self._expect(JsTokenKind.COLON)
        elif self._eat(JsTokenKind.DEFAULT):
            self._expect(JsTokenKind.COLON)
        else:
            self._advance()
        body: list[Statement] = []
        while not self._at(
            JsTokenKind.CASE, JsTokenKind.DEFAULT, JsTokenKind.RBRACE, JsTokenKind.EOF,
        ):
            stmt = self._parse_statement()
            if stmt is not None:
                body.append(stmt)
        return JsSwitchCase(test=test, body=body, offset=offset)

    def _parse_try_statement(self) -> JsTryStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.TRY)
        block = self._parse_block_statement()
        handler = None
        finalizer = None
        if self._eat(JsTokenKind.CATCH):
            handler = self._parse_catch_clause()
        if self._eat(JsTokenKind.FINALLY):
            finalizer = self._parse_block_statement()
        return JsTryStatement(
            block=block, handler=handler, finalizer=finalizer, offset=offset)

    def _parse_catch_clause(self) -> JsCatchClause:
        offset = self._current.offset
        param = None
        if self._eat(JsTokenKind.LPAREN):
            param = self._parse_binding_pattern()
            self._expect(JsTokenKind.RPAREN)
        body = self._parse_block_statement()
        return JsCatchClause(param=param, body=body, offset=offset)

    def _parse_with_statement(self) -> JsWithStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.WITH)
        self._expect(JsTokenKind.LPAREN)
        obj = self._parse_expression()
        self._expect(JsTokenKind.RPAREN)
        body = self._parse_statement()
        return JsWithStatement(object=obj, body=body, offset=offset)

    def _parse_return_statement(self) -> JsReturnStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.RETURN)
        argument = None
        if not self._preceded_by_newline and not self._at(
            JsTokenKind.SEMICOLON, JsTokenKind.RBRACE, JsTokenKind.EOF,
        ):
            argument = self._parse_expression()
        self._eat_semicolon()
        return JsReturnStatement(argument=argument, offset=offset)

    def _parse_throw_statement(self) -> JsThrowStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.THROW)
        argument = None
        if not self._preceded_by_newline:
            argument = self._parse_expression()
        self._eat_semicolon()
        return JsThrowStatement(argument=argument, offset=offset)

    def _parse_break_statement(self) -> JsBreakStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.BREAK)
        label = None
        if not self._preceded_by_newline and self._at_binding_identifier():
            tok = self._advance()
            label = JsIdentifier(name=tok.value, offset=tok.offset)
        self._eat_semicolon()
        return JsBreakStatement(label=label, offset=offset)

    def _parse_continue_statement(self) -> JsContinueStatement:
        offset = self._current.offset
        self._expect(JsTokenKind.CONTINUE)
        label = None
        if not self._preceded_by_newline and self._at_binding_identifier():
            tok = self._advance()
            label = JsIdentifier(name=tok.value, offset=tok.offset)
        self._eat_semicolon()
        return JsContinueStatement(label=label, offset=offset)

    def _parse_function_impl(
        self,
        *,
        as_expression: bool,
        is_async: bool = False,
    ) -> JsFunctionDeclaration | JsFunctionExpression:
        offset = self._current.offset
        self._expect(JsTokenKind.FUNCTION)
        generator = bool(self._eat(JsTokenKind.STAR))
        id_node = None
        if self._at_binding_identifier():
            tok = self._advance()
            id_node = JsIdentifier(name=tok.value, offset=tok.offset)
        with self._function_body_context(is_async, generator):
            params = self._parse_formal_parameters()
            body = self._parse_block_statement()
        if as_expression:
            return JsFunctionExpression(
                id=id_node, params=params, body=body,
                generator=generator, is_async=is_async, offset=offset)
        return JsFunctionDeclaration(
            id=id_node, params=params, body=body,
            generator=generator, is_async=is_async, offset=offset)

    def _parse_function_declaration(
        self,
        is_async: bool = False,
    ) -> JsFunctionDeclaration:
        return self._parse_function_impl(as_expression=False, is_async=is_async)

    def _parse_formal_parameters(self) -> list[Expression]:
        self._expect(JsTokenKind.LPAREN)
        params: list[Expression] = []
        while not self._at(JsTokenKind.RPAREN, JsTokenKind.EOF):
            if self._at(JsTokenKind.ELLIPSIS):
                params.append(self._parse_rest_element())
                break
            param = self._parse_binding_pattern()
            if self._eat(JsTokenKind.EQUALS):
                default = self._parse_assignment_expression()
                param = JsAssignmentPattern(
                    left=param, right=default, offset=param.offset)
            params.append(param)
            if not self._at(JsTokenKind.RPAREN):
                self._expect(JsTokenKind.COMMA)
        self._expect(JsTokenKind.RPAREN)
        return params

    def _parse_decorators(self) -> list[JsDecorator]:
        decorators: list[JsDecorator] = []
        while self._at(JsTokenKind.AT):
            decorators.append(self._parse_decorator())
        return decorators

    def _parse_decorator(self) -> JsDecorator:
        offset = self._current.offset
        self._expect(JsTokenKind.AT)
        if self._at(JsTokenKind.LPAREN):
            self._advance()
            inner = self._parse_expression()
            self._expect(JsTokenKind.RPAREN)
            return JsDecorator(expression=inner, offset=offset)
        if not self._at_binding_identifier():
            return JsDecorator(
                expression=JsErrorNode(
                    text=self._current.value, message='unexpected token', offset=offset),
                offset=offset,
            )
        tok = self._advance()
        expr: Expression = JsIdentifier(name=tok.value, offset=tok.offset)
        while self._eat(JsTokenKind.DOT):
            prop = self._advance()
            expr = JsMemberExpression(
                object=expr,
                property=JsIdentifier(name=prop.value, offset=prop.offset),
                computed=False,
                offset=expr.offset,
            )
        if self._at(JsTokenKind.LPAREN):
            expr = self._parse_call_arguments(expr, optional=False)
        return JsDecorator(expression=expr, offset=offset)

    def _parse_class_impl(
        self,
        *,
        as_expression: bool,
        decorators: list[JsDecorator] | None = None,
    ) -> JsClassDeclaration | JsClassExpression:
        offset = self._current.offset
        self._expect(JsTokenKind.CLASS)
        id_node = None
        if self._at_binding_identifier():
            tok = self._advance()
            id_node = JsIdentifier(name=tok.value, offset=tok.offset)
        super_class = None
        if self._eat(JsTokenKind.EXTENDS):
            super_class = self._parse_assignment_expression()
        body = self._parse_class_body()
        if as_expression:
            return JsClassExpression(
                id=id_node,
                super_class=super_class,
                body=body,
                decorators=decorators or [],
                offset=offset,
            )
        return JsClassDeclaration(
            id=id_node,
            super_class=super_class,
            body=body,
            decorators=decorators or [],
            offset=offset,
        )

    def _parse_class_declaration(
        self, decorators: list[JsDecorator] | None = None,
    ) -> JsClassDeclaration:
        return self._parse_class_impl(as_expression=False, decorators=decorators)

    def _parse_class_body(self) -> JsClassBody:
        offset = self._current.offset
        self._expect(JsTokenKind.LBRACE)
        members: list[JsMethodDefinition | JsPropertyDefinition | JsStaticBlock] = []
        while not self._at(JsTokenKind.RBRACE, JsTokenKind.EOF):
            if self._eat(JsTokenKind.SEMICOLON):
                continue
            decorators = self._parse_decorators()
            member = self._parse_class_member()
            if decorators and isinstance(member, (JsMethodDefinition, JsPropertyDefinition)):
                member.decorators = decorators
                member._adopt(*decorators)
            members.append(member)
        self._expect(JsTokenKind.RBRACE)
        return JsClassBody(body=members, offset=offset)

    def _parse_static_block(self, offset: int) -> JsStaticBlock:
        block = self._parse_block_statement()
        return JsStaticBlock(body=block.body, offset=offset)

    def _parse_class_member(self) -> JsMethodDefinition | JsPropertyDefinition | JsStaticBlock:
        offset = self._current.offset
        is_static = False
        if self._at(JsTokenKind.IDENTIFIER) and self._current.value == 'static':
            saved_pos = self._current
            self._advance()
            if self._at(JsTokenKind.LBRACE):
                return self._parse_static_block(offset)
            if self._at(JsTokenKind.LPAREN):
                key = JsIdentifier(name='static', offset=saved_pos.offset)
                return self._finish_class_member(key, False, False, offset)
            if self._at_class_field_terminator():
                key = JsIdentifier(name='static', offset=saved_pos.offset)
                return self._finish_class_field(key, False, False, offset)
            is_static = True

        kind = JsMethodKind.METHOD
        is_generator = bool(self._eat(JsTokenKind.STAR))
        is_async = False

        if (
            not is_generator
            and self._at(JsTokenKind.IDENTIFIER)
            and self._current.value in ('get', 'set')
        ):
            saved = self._current
            self._advance()
            if self._at(JsTokenKind.LPAREN):
                key = JsIdentifier(name=saved.value, offset=saved.offset)
                return self._finish_class_member(key, is_static, False, offset)
            if self._at_class_field_terminator():
                key = JsIdentifier(name=saved.value, offset=saved.offset)
                return self._finish_class_field(key, is_static, False, offset)
            kind = JsMethodKind.GET if saved.value == 'get' else JsMethodKind.SET
        elif not is_generator and self._at(JsTokenKind.ASYNC):
            saved = self._current
            self._advance()
            if self._at(JsTokenKind.LPAREN):
                key = JsIdentifier(name='async', offset=saved.offset)
                return self._finish_class_member(key, is_static, False, offset)
            if self._preceded_by_newline or self._at_class_field_terminator():
                key = JsIdentifier(name='async', offset=saved.offset)
                return self._finish_class_field(key, is_static, False, offset)
            is_async = True
            if self._eat(JsTokenKind.STAR):
                is_generator = True

        key, computed = self._parse_property_key()

        if kind == JsMethodKind.METHOD and not is_generator and not self._at(JsTokenKind.LPAREN):
            return self._finish_class_field(key, is_static, computed, offset)

        return self._finish_class_member(key, is_static, is_generator, offset, kind, computed, is_async=is_async)

    def _at_class_field_terminator(self) -> bool:
        """
        Whether the current token completes a class element as a field named by the identifier just consumed:
        an initializer (`=`), an explicit terminator (`;`), or the end of the class body (`}` / end of input).
        A modifier prefix (`static`/`get`/`set`/`async`) followed by one of these is an ordinary field whose
        name happens to be that word, not a modifier.
        """
        return self._at(
            JsTokenKind.EQUALS,
            JsTokenKind.SEMICOLON,
            JsTokenKind.RBRACE,
            JsTokenKind.EOF,
        )

    def _finish_class_field(
        self,
        key: Expression,
        is_static: bool,
        computed: bool,
        offset: int,
    ) -> JsPropertyDefinition:
        value = None
        if self._eat(JsTokenKind.EQUALS):
            value = self._parse_assignment_expression()
        self._eat_semicolon()
        return JsPropertyDefinition(
            key=key,
            value=value,
            computed=computed,
            is_static=is_static,
            offset=offset,
        )

    def _finish_class_member(
        self,
        key: Expression,
        is_static: bool,
        is_generator: bool,
        offset: int,
        kind: JsMethodKind = JsMethodKind.METHOD,
        computed: bool = False,
        is_async: bool = False,
    ) -> JsMethodDefinition:
        func_offset = self._current.offset
        with self._function_body_context(is_async, is_generator):
            params = self._parse_formal_parameters()
            body = self._parse_block_statement()
        value = JsFunctionExpression(
            params=params,
            body=body,
            generator=is_generator,
            is_async=is_async,
            offset=func_offset,
        )
        if isinstance(key, JsIdentifier) and key.name == 'constructor' and kind == JsMethodKind.METHOD:
            kind = JsMethodKind.CONSTRUCTOR
        return JsMethodDefinition(
            key=key,
            value=value,
            kind=kind,
            computed=computed,
            is_static=is_static,
            offset=offset,
        )

    def _parse_import_expression(self, offset: int) -> Expression:
        self._expect(JsTokenKind.IMPORT)
        if self._eat(JsTokenKind.DOT):
            prop = self._advance()
            return JsMetaProperty(meta='import', property=prop.value, offset=offset)
        if self._at(JsTokenKind.LPAREN):
            self._advance()
            source = self._parse_assignment_expression()
            options = None
            if self._eat(JsTokenKind.COMMA) and not self._at(JsTokenKind.RPAREN):
                options = self._parse_assignment_expression()
                self._eat(JsTokenKind.COMMA)
            self._expect(JsTokenKind.RPAREN)
            return JsImportExpression(source=source, options=options, offset=offset)
        return JsErrorNode(text='import', message='unexpected token', offset=offset)

    def _parse_import_attributes(self) -> tuple[str, list[JsImportAttribute]]:
        if self._preceded_by_newline:
            return '', []
        if self._at(JsTokenKind.WITH):
            keyword = 'with'
        elif self._at(JsTokenKind.IDENTIFIER) and self._current.value == 'assert':
            keyword = 'assert'
        else:
            return '', []
        self._advance()
        attributes: list[JsImportAttribute] = []
        self._expect(JsTokenKind.LBRACE)
        while not self._at(JsTokenKind.RBRACE, JsTokenKind.EOF):
            key = self._parse_property_name()
            self._expect(JsTokenKind.COLON)
            value = self._parse_string_literal()
            attributes.append(JsImportAttribute(key=key, value=value, offset=key.offset))
            if not self._eat(JsTokenKind.COMMA):
                break
        self._expect(JsTokenKind.RBRACE)
        return keyword, attributes

    def _module_specifier(self) -> JsStringLiteral | None:
        """
        The literal naming the module a declaration reads from, or `None` where none stands there.
        It is the one part of these declarations the grammar gives no default for, so a source that
        ends before writing it has not written the declaration at all; answering with a literal
        spelled by nothing states a module whose name is the empty string, which is a module the
        file could have named and did not.
        """
        if self._at(JsTokenKind.STRING_SINGLE, JsTokenKind.STRING_DOUBLE):
            return self._parse_string_literal()
        return None

    def _unread_since(self, offset: int, message: str) -> JsErrorNode:
        """
        The source from *offset* up to where reading stands, handed back as itself. A declaration
        the parser could not complete is kept whole rather than in the parts it did manage to read:
        what prints is then what was written, and reading that print again finds the same thing,
        where a half-built declaration prints the halves it has and reads back as something else.
        """
        return JsErrorNode(
            text=self._source[offset:self._current.offset].rstrip(),
            message=message,
            offset=offset,
        )

    def _parse_import_declaration(self) -> JsImportDeclaration | JsErrorNode:
        offset = self._current.offset
        self._expect(JsTokenKind.IMPORT)

        if self._at(JsTokenKind.STRING_SINGLE, JsTokenKind.STRING_DOUBLE):
            source = self._parse_string_literal()
            keyword, attributes = self._parse_import_attributes()
            self._eat_semicolon()
            return JsImportDeclaration(
                source=source, attributes=attributes, attributes_keyword=keyword, offset=offset)

        specifiers: list[
            JsImportSpecifier | JsImportDefaultSpecifier | JsImportNamespaceSpecifier
        ] = []

        if self._at_binding_identifier():
            tok = self._advance()
            specifiers.append(JsImportDefaultSpecifier(
                local=JsIdentifier(name=tok.value, offset=tok.offset),
                offset=tok.offset,
            ))
            if self._eat(JsTokenKind.COMMA):
                if self._at(JsTokenKind.STAR):
                    specifiers.append(self._parse_namespace_import())
                elif self._at(JsTokenKind.LBRACE):
                    specifiers.extend(self._parse_named_imports())

        elif self._at(JsTokenKind.STAR):
            specifiers.append(self._parse_namespace_import())

        elif self._at(JsTokenKind.LBRACE):
            specifiers.extend(self._parse_named_imports())

        self._expect_contextual('from')
        source = self._module_specifier()
        if source is None:
            return self._unread_since(offset, 'a module declaration with no specifier')
        keyword, attributes = self._parse_import_attributes()
        self._eat_semicolon()
        return JsImportDeclaration(
            specifiers=specifiers,
            source=source,
            attributes=attributes,
            attributes_keyword=keyword,
            offset=offset,
        )

    def _parse_namespace_import(self) -> JsImportNamespaceSpecifier:
        offset = self._current.offset
        self._expect(JsTokenKind.STAR)
        self._expect_contextual('as')
        tok = self._expect(JsTokenKind.IDENTIFIER)
        return JsImportNamespaceSpecifier(
            local=self._name_or_error(tok.value, tok.offset),
            offset=offset,
        )

    def _parse_named_imports(self) -> list[JsImportSpecifier]:
        self._expect(JsTokenKind.LBRACE)
        specs: list[JsImportSpecifier] = []
        while not self._at(JsTokenKind.RBRACE, JsTokenKind.EOF):
            spec_offset = self._current.offset
            tok = self._advance()
            imported = self._name_or_error(tok.value, tok.offset)
            local = imported
            if self._at(JsTokenKind.AS):
                self._advance()
                ltok = self._expect(JsTokenKind.IDENTIFIER)
                local = self._name_or_error(ltok.value, ltok.offset)
            specs.append(JsImportSpecifier(
                imported=imported, local=local, offset=spec_offset))
            if not self._at(JsTokenKind.RBRACE):
                self._expect(JsTokenKind.COMMA)
        self._expect(JsTokenKind.RBRACE)
        return specs

    def _parse_export_declaration(
        self, decorators: list[JsDecorator] | None = None,
    ) -> Statement:
        offset = self._current.offset
        self._expect(JsTokenKind.EXPORT)
        class_decorators = list(decorators or []) + self._parse_decorators()

        if self._eat(JsTokenKind.DEFAULT):
            class_decorators += self._parse_decorators()
            if self._at(JsTokenKind.FUNCTION):
                decl = self._parse_function_declaration()
                return JsExportDefaultDeclaration(declaration=decl, offset=offset)
            if self._at(JsTokenKind.CLASS):
                decl = self._parse_class_declaration(class_decorators)
                return JsExportDefaultDeclaration(declaration=decl, offset=offset)
            if self._at_async_function():
                self._advance()
                decl = self._parse_function_declaration(is_async=True)
                return JsExportDefaultDeclaration(declaration=decl, offset=offset)
            expr = self._parse_assignment_expression()
            self._eat_semicolon()
            return JsExportDefaultDeclaration(declaration=expr, offset=offset)

        if self._at(JsTokenKind.STAR):
            self._advance()
            exported = None
            if self._at(JsTokenKind.AS):
                self._advance()
                tok = self._expect(JsTokenKind.IDENTIFIER)
                exported = self._name_or_error(tok.value, tok.offset)
            self._expect_contextual('from')
            source = self._module_specifier()
            if source is None:
                return self._unread_since(offset, 'a module declaration with no specifier')
            self._eat_semicolon()
            return JsExportAllDeclaration(
                source=source, exported=exported, offset=offset)

        if self._at(JsTokenKind.LBRACE):
            return self._parse_export_named(offset)

        if self._at(JsTokenKind.VAR, JsTokenKind.LET, JsTokenKind.CONST):
            decl = self._parse_variable_declaration()
            return JsExportNamedDeclaration(declaration=decl, offset=offset)
        if self._at(JsTokenKind.FUNCTION):
            decl = self._parse_function_declaration()
            return JsExportNamedDeclaration(declaration=decl, offset=offset)
        if self._at(JsTokenKind.CLASS):
            decl = self._parse_class_declaration(class_decorators)
            return JsExportNamedDeclaration(declaration=decl, offset=offset)
        if self._at_async_function():
            self._advance()
            decl = self._parse_function_declaration(is_async=True)
            return JsExportNamedDeclaration(declaration=decl, offset=offset)

        self._advance()
        return JsExportNamedDeclaration(offset=offset)

    def _parse_export_named(self, offset: int) -> JsExportNamedDeclaration | JsErrorNode:
        self._expect(JsTokenKind.LBRACE)
        specifiers: list[JsExportSpecifier] = []
        while not self._at(JsTokenKind.RBRACE, JsTokenKind.EOF):
            spec_offset = self._current.offset
            tok = self._advance()
            local = self._name_or_error(tok.value, tok.offset)
            exported = local
            if self._at(JsTokenKind.AS):
                self._advance()
                etok = self._advance()
                exported = self._name_or_error(etok.value, etok.offset)
            specifiers.append(JsExportSpecifier(
                local=local, exported=exported, offset=spec_offset))
            if not self._at(JsTokenKind.RBRACE):
                self._expect(JsTokenKind.COMMA)
        self._expect(JsTokenKind.RBRACE)
        source = None
        if self._at(JsTokenKind.FROM):
            self._advance()
            source = self._module_specifier()
            if source is None:
                return self._unread_since(offset, 'a module declaration with no specifier')
        self._eat_semicolon()
        return JsExportNamedDeclaration(
            specifiers=specifiers, source=source, offset=offset)

    def _at_async_function(self) -> bool:
        """
        Whether the parser is positioned at `async function` with no line terminator between the two — the
        one form in which a leading `async` opens a declaration rather than an ordinary expression. Every
        other `async` (a call, member access, arrow, or bare reference) is left to the expression grammar,
        which reaches it through `_parse_async_expression` and applies the full call/member and operator
        parsing.
        """
        return (
            self._at(JsTokenKind.ASYNC)
            and self._peek_next().kind == JsTokenKind.FUNCTION
            and not self._ahead_newline
        )

    def _expect_contextual(self, keyword: str):
        if self._at(JsTokenKind.FROM) and keyword == 'from':
            self._advance()
            return
        if self._at(JsTokenKind.AS) and keyword == 'as':
            self._advance()
            return
        if self._at(JsTokenKind.IDENTIFIER) and self._current.value == keyword:
            self._advance()
            return
        self._advance()

    def _parse_expression(self) -> Expression:
        expr = self._parse_assignment_expression()
        if self._at(JsTokenKind.COMMA):
            exprs = [expr]
            while self._eat(JsTokenKind.COMMA):
                exprs.append(self._parse_assignment_expression())
            return JsSequenceExpression(expressions=exprs, offset=expr.offset)
        return expr

    def _parse_assignment_expression(self) -> Expression:
        """
        An AssignmentExpression, which is the only production a YieldExpression is one of. Reading
        the `yield` here rather than among the primary expressions is what stops an operator from
        attaching to it: a `yield` that the line terminator restriction left without an argument
        ends the expression, and the slash that opens the next statement is not its divisor.
        """
        if self._at(JsTokenKind.YIELD) and self._in_generator:
            return self._parse_yield_expression()
        left = self._parse_conditional_expression()
        if self._current.kind.is_assignment:
            op = self._advance().value
            right = self._parse_assignment_expression()
            left = self._to_param(left) if op == '=' else left
            return JsAssignmentExpression(
                left=left, operator=op, right=right, offset=left.offset)
        return left

    def _parse_conditional_expression(self) -> Expression:
        expr = self._parse_binary_expression()
        if self._eat(JsTokenKind.QUESTION):
            consequent = self._parse_assignment_expression()
            self._expect(JsTokenKind.COLON)
            alternate = self._parse_assignment_expression()
            return JsConditionalExpression(
                test=expr,
                consequent=consequent,
                alternate=alternate,
                offset=expr.offset,
            )
        return expr

    def _parse_binary_expression(self, min_prec: int = 0) -> Expression:
        left = self._parse_unary_expression()
        while True:
            entry = _BINARY_PREC.get(self._current.kind)
            if entry is None:
                break
            prec, logical = entry
            if prec < min_prec:
                break
            if self._no_in and self._at(JsTokenKind.IN):
                break
            op = self._advance().value
            next_prec = prec if prec == _PREC_EXPONENTIATION else prec + 1
            right = self._parse_binary_expression(next_prec)
            node_type = JsLogicalExpression if logical else JsBinaryExpression
            left = node_type(
                left=left, operator=op, right=right, offset=left.offset)
        return left

    def _parse_unary_expression(self) -> Expression:
        if self._at(
            JsTokenKind.BANG,
            JsTokenKind.TILDE,
            JsTokenKind.TYPEOF,
            JsTokenKind.VOID,
            JsTokenKind.DELETE,
        ):
            tok = self._advance()
            operand = self._parse_unary_expression()
            return JsUnaryExpression(
                operator=tok.value, operand=operand, prefix=True, offset=tok.offset)
        if self._at(JsTokenKind.PLUS):
            tok = self._advance()
            operand = self._parse_unary_expression()
            return JsUnaryExpression(
                operator='+', operand=operand, prefix=True, offset=tok.offset)
        if self._at(JsTokenKind.MINUS):
            tok = self._advance()
            operand = self._parse_unary_expression()
            return JsUnaryExpression(
                operator='-', operand=operand, prefix=True, offset=tok.offset)
        if self._at(JsTokenKind.AWAIT) and self._in_async:
            tok = self._advance()
            operand = self._parse_unary_expression()
            return JsAwaitExpression(argument=operand, offset=tok.offset)
        return self._parse_update_expression()

    def _parse_update_expression(self) -> Expression:
        if self._at(JsTokenKind.INC, JsTokenKind.DEC):
            tok = self._advance()
            argument = self._parse_call_expression()
            return JsUpdateExpression(
                operator=tok.value, argument=argument, prefix=True, offset=tok.offset)
        expr = self._parse_call_expression()
        if not self._preceded_by_newline and self._at(
            JsTokenKind.INC, JsTokenKind.DEC,
        ):
            tok = self._advance()
            return JsUpdateExpression(
                operator=tok.value, argument=expr, prefix=False, offset=expr.offset)
        return expr

    def _parse_call_expression(self) -> Expression:
        """
        A left-hand side expression: what may be called, indexed, or used to tag a template. An
        arrow function is none of those. It is an AssignmentExpression and never a
        LeftHandSideExpression, so nothing may attach to it, and the tail that would have attached
        belongs to whatever follows instead — `f = a => {}` on one line and `[x].forEach(g)` on the
        next are two statements, and reading the bracket as an index into the arrow makes them one.
        """
        expr = self._parse_new_expression()
        if isinstance(expr, JsArrowFunctionExpression):
            return expr
        while True:
            if self._at(JsTokenKind.LPAREN):
                expr = self._parse_call_arguments(expr, optional=False)
            elif self._eat(JsTokenKind.DOT):
                prop = self._member_property()
                expr = JsMemberExpression(
                    object=expr, property=prop, computed=False, optional=False, offset=expr.offset)
            elif self._at(JsTokenKind.LBRACKET):
                self._advance()
                prop = self._parse_expression()
                self._expect(JsTokenKind.RBRACKET)
                expr = JsMemberExpression(
                    object=expr, property=prop, computed=True, optional=False, offset=expr.offset)
            elif self._eat(JsTokenKind.QUESTION_DOT):
                if self._at(JsTokenKind.LPAREN):
                    expr = self._parse_call_arguments(expr, optional=True)
                elif self._at(JsTokenKind.LBRACKET):
                    self._advance()
                    prop = self._parse_expression()
                    self._expect(JsTokenKind.RBRACKET)
                    expr = JsMemberExpression(
                        object=expr, property=prop, computed=True, optional=True, offset=expr.offset)
                else:
                    prop = self._member_property()
                    expr = JsMemberExpression(
                        object=expr, property=prop, computed=False, optional=True, offset=expr.offset)
            elif self._at(
                JsTokenKind.TEMPLATE_FULL, JsTokenKind.TEMPLATE_HEAD,
            ):
                quasi = self._parse_template_literal()
                expr = JsTaggedTemplateExpression(
                    tag=expr, quasi=quasi, offset=expr.offset)
            else:
                break
        return expr

    def _member_property(self) -> Expression:
        """
        The name behind a dot. It is an IdentifierName, which is every word the language has and not
        only the ones that may be a variable, so `a.if` and `a.default` are ordinary member reads.

        Where the text behind the dot spells no word at all there is no name to build. An identifier
        with no name spells nothing — printing it writes the dot and stops, which is not a program —
        so what was read is handed back as itself instead.
        """
        tok = self._advance()
        if tok.kind is JsTokenKind.PRIVATE_IDENTIFIER:
            return JsPrivateIdentifier(name=tok.value[1:], offset=tok.offset)
        if tok.kind is JsTokenKind.IDENTIFIER or tok.kind.is_keyword:
            return JsIdentifier(name=tok.value, offset=tok.offset)
        return JsErrorNode(
            text=tok.value, message='expected a property name', offset=tok.offset)

    def _parse_argument_list(self) -> list[Expression]:
        args: list[Expression] = []
        while not self._at(JsTokenKind.RPAREN, JsTokenKind.EOF):
            if self._at(JsTokenKind.ELLIPSIS):
                offset = self._current.offset
                self._advance()
                arg = self._parse_assignment_expression()
                args.append(JsSpreadElement(argument=arg, offset=offset))
            else:
                args.append(self._parse_assignment_expression())
            if not self._at(JsTokenKind.RPAREN):
                self._expect(JsTokenKind.COMMA)
        self._expect(JsTokenKind.RPAREN)
        return args

    def _parse_call_arguments(
        self,
        callee: Expression,
        optional: bool,
    ) -> JsCallExpression:
        with self._with_no_in(False):
            self._expect(JsTokenKind.LPAREN)
            args = self._parse_argument_list()
        return JsCallExpression(
            callee=callee, arguments=args, optional=optional, offset=callee.offset)

    def _parse_new_expression(self) -> Expression:
        if self._at(JsTokenKind.NEW):
            offset = self._current.offset
            self._advance()
            if self._at(JsTokenKind.DOT):
                self._advance()
                return JsMemberExpression(
                    object=JsIdentifier(name='new', offset=offset),
                    property=self._member_property(),
                    computed=False,
                    offset=offset,
                )
            if self._at(JsTokenKind.ASYNC) and not self._at_async_function():
                tok = self._advance()
                callee = JsIdentifier(name=tok.value, offset=tok.offset)
            else:
                callee = self._parse_new_expression()
            while True:
                if self._eat(JsTokenKind.DOT):
                    prop = self._member_property()
                    callee = JsMemberExpression(
                        object=callee, property=prop, computed=False, offset=callee.offset)
                elif self._at(JsTokenKind.LBRACKET):
                    self._advance()
                    prop = self._parse_expression()
                    self._expect(JsTokenKind.RBRACKET)
                    callee = JsMemberExpression(
                        object=callee, property=prop, computed=True, offset=callee.offset)
                else:
                    break
            args: list[Expression] = []
            if self._at(JsTokenKind.LPAREN):
                self._advance()
                args = self._parse_argument_list()
            return JsNewExpression(callee=callee, arguments=args, offset=offset)
        return self._parse_primary_expression()

    def _parse_primary_expression(self) -> Expression:
        tok = self._current
        offset = tok.offset

        if self._at(JsTokenKind.ASYNC):
            return self._parse_async_expression()

        if self._at_binding_identifier():
            self._advance()
            if self._at(JsTokenKind.ARROW) and not self._preceded_by_newline:
                self._advance()
                param = JsIdentifier(name=tok.value, offset=offset)
                body = self._parse_arrow_body()
                return JsArrowFunctionExpression(
                    params=[param], body=body, offset=offset)
            return JsIdentifier(name=tok.value, offset=offset)

        if self._at(JsTokenKind.PRIVATE_IDENTIFIER):
            self._advance()
            return JsPrivateIdentifier(name=tok.value[1:], offset=offset)

        if self._at(JsTokenKind.IMPORT):
            return self._parse_import_expression(offset)

        if self._at(JsTokenKind.INTEGER):
            self._advance()
            raw = tok.value
            value = self._parse_int_text(raw.replace('_', ''))
            return JsNumericLiteral(value=value, raw=raw, offset=offset)

        if self._at(JsTokenKind.FLOAT):
            self._advance()
            raw = tok.value
            value = float(raw.replace('_', ''))
            return JsNumericLiteral(value=value, raw=raw, offset=offset)

        if self._at(JsTokenKind.BIGINT):
            self._advance()
            raw = tok.value
            value = self._parse_int_text(raw.replace('_', '').rstrip('n'))
            return JsBigIntLiteral(value=value, raw=raw, offset=offset)

        if self._at(JsTokenKind.STRING_SINGLE, JsTokenKind.STRING_DOUBLE):
            return self._parse_string_literal()

        if self._at(JsTokenKind.SLASH, JsTokenKind.SLASH_ASSIGN):
            tok = self._rescan_as_regexp()

        if self._at(JsTokenKind.REGEXP):
            self._advance()
            raw = tok.value
            last_slash = raw.rfind('/')
            pattern = raw[1:last_slash]
            flags = raw[last_slash + 1:]
            return JsRegExpLiteral(
                pattern=pattern, flags=flags, raw=raw, offset=offset)

        if self._at(JsTokenKind.TEMPLATE_FULL, JsTokenKind.TEMPLATE_HEAD):
            return self._parse_template_literal()

        if self._at(JsTokenKind.TRUE):
            self._advance()
            return JsBooleanLiteral(value=True, offset=offset)
        if self._at(JsTokenKind.FALSE):
            self._advance()
            return JsBooleanLiteral(value=False, offset=offset)
        if self._at(JsTokenKind.NULL):
            self._advance()
            return JsNullLiteral(offset=offset)
        if self._at(JsTokenKind.THIS):
            self._advance()
            return JsThisExpression(offset=offset)
        if self._at(JsTokenKind.SUPER):
            self._advance()
            return JsIdentifier(name='super', offset=offset)

        if self._at(JsTokenKind.LBRACKET):
            return self._parse_array_literal()
        if self._at(JsTokenKind.LBRACE):
            return self._parse_object_literal()

        if self._at(JsTokenKind.LPAREN):
            return self._parse_paren_or_arrow()

        if self._at(JsTokenKind.FUNCTION):
            return self._parse_function_expression()
        if self._at(JsTokenKind.CLASS):
            return self._parse_class_expression()

        self._advance()
        return JsErrorNode(text=tok.value, message='unexpected token', offset=offset)

    def _parse_string_literal(self) -> JsStringLiteral:
        tok = self._advance()
        raw = tok.value
        end = len(raw) - 1 if tok.terminated else len(raw)
        return JsStringLiteral(
            value=decode_js_string_body(raw[1:end]),
            raw=raw,
            terminated=tok.terminated,
            offset=tok.offset,
        )

    @staticmethod
    def _template_element(tok: JsToken, tail: bool) -> JsTemplateElement:
        """
        One run of text of a template literal, taken from the token without the delimiters around
        it: one character opens every run, and the one that ends it is a backtick where the run
        ends the literal and `${` where a hole follows. The text between them is cooked into what
        it denotes, exactly as the body of a string literal is — a template that carries an escape
        means what the escape means, and reading it as the characters that spell it is how a `\\t`
        became two.
        """
        raw = tok.value
        end = len(raw) - (1 if tail else 2) if tok.terminated else len(raw)
        text = raw[1:end]
        return JsTemplateElement(
            value=decode_js_template_body(text),
            raw=text,
            tail=tail,
            terminated=tok.terminated,
            offset=tok.offset,
        )

    def _parse_template_literal(self) -> JsTemplateLiteral:
        offset = self._current.offset
        quasis: list[JsTemplateElement] = []
        expressions: list[Expression] = []

        if self._at(JsTokenKind.TEMPLATE_FULL):
            quasis.append(self._template_element(self._advance(), True))
            return JsTemplateLiteral(
                quasis=quasis, expressions=expressions, offset=offset)

        quasis.append(self._template_element(self._advance(), False))

        while True:
            expressions.append(self._parse_expression())
            if self._at(JsTokenKind.TEMPLATE_TAIL):
                quasis.append(self._template_element(self._advance(), True))
                break
            elif self._at(JsTokenKind.TEMPLATE_MIDDLE):
                quasis.append(self._template_element(self._advance(), False))
            else:
                quasis.append(JsTemplateElement(
                    value='',
                    raw='',
                    tail=True,
                    terminated=False,
                    offset=self._current.offset,
                ))
                break

        return JsTemplateLiteral(
            quasis=quasis, expressions=expressions, offset=offset)

    def _parse_array_literal(self) -> JsArrayExpression:
        with self._with_no_in(False):
            offset = self._current.offset
            self._expect(JsTokenKind.LBRACKET)
            elements: list[Expression | None] = []
            while not self._at(JsTokenKind.RBRACKET, JsTokenKind.EOF):
                if self._at(JsTokenKind.COMMA):
                    elements.append(None)
                    self._advance()
                    continue
                if self._at(JsTokenKind.ELLIPSIS):
                    so = self._current.offset
                    self._advance()
                    arg = self._parse_assignment_expression()
                    elements.append(JsSpreadElement(argument=arg, offset=so))
                else:
                    elements.append(self._parse_assignment_expression())
                if not self._at(JsTokenKind.RBRACKET):
                    self._eat(JsTokenKind.COMMA)
            self._expect(JsTokenKind.RBRACKET)
        return JsArrayExpression(elements=elements, offset=offset)

    def _parse_object_literal(self) -> JsObjectExpression:
        with self._with_no_in(False):
            offset = self._current.offset
            self._expect(JsTokenKind.LBRACE)
            properties: list[JsProperty | JsSpreadElement] = []
            while not self._at(JsTokenKind.RBRACE, JsTokenKind.EOF):
                if self._at(JsTokenKind.ELLIPSIS):
                    so = self._current.offset
                    self._advance()
                    arg = self._parse_assignment_expression()
                    properties.append(JsSpreadElement(argument=arg, offset=so))
                else:
                    properties.append(self._parse_object_property())
                if not self._at(JsTokenKind.RBRACE):
                    self._eat(JsTokenKind.COMMA)
            self._expect(JsTokenKind.RBRACE)
        return JsObjectExpression(properties=properties, offset=offset)

    def _parse_object_property(self) -> JsProperty:
        offset = self._current.offset
        is_generator = bool(self._eat(JsTokenKind.STAR))

        if (
            self._at(JsTokenKind.IDENTIFIER)
            and self._current.value in ('get', 'set')
            and not is_generator
        ):
            kind_val = _PROP_KIND_MAP[self._current.value]
            saved = self._current
            self._advance()
            if self._at(JsTokenKind.LPAREN):
                key = JsIdentifier(name=saved.value, offset=saved.offset)
                return self._make_method_property(key, JsPropertyKind.INIT, False, offset)
            if self._at(
                JsTokenKind.COLON,
                JsTokenKind.COMMA,
                JsTokenKind.RBRACE,
                JsTokenKind.EQUALS,
            ):
                key = JsIdentifier(name=saved.value, offset=saved.offset)
                return self._finish_property_value(key, False, offset)
            key, computed = self._parse_property_key()
            return self._make_method_property(key, kind_val, False, offset, computed=computed)

        if self._at(JsTokenKind.ASYNC) and not is_generator:
            saved = self._current
            self._advance()
            if self._preceded_by_newline or self._at(
                JsTokenKind.COLON,
                JsTokenKind.COMMA,
                JsTokenKind.RBRACE,
                JsTokenKind.EQUALS,
                JsTokenKind.LPAREN,
            ):
                if self._at(JsTokenKind.LPAREN):
                    key = JsIdentifier(name='async', offset=saved.offset)
                    return self._make_method_property(key, JsPropertyKind.INIT, False, offset)
                key = JsIdentifier(name='async', offset=saved.offset)
                return self._finish_property_value(key, False, offset)
            gen = bool(self._eat(JsTokenKind.STAR))
            key, computed = self._parse_property_key()
            return self._make_method_property(
                key, JsPropertyKind.INIT, gen, offset, computed=computed, is_async=True)

        key, computed = self._parse_property_key()

        if is_generator or self._at(JsTokenKind.LPAREN):
            return self._make_method_property(key, JsPropertyKind.INIT, is_generator, offset, computed=computed)

        return self._finish_property_value(key, computed, offset)

    def _finish_property_value(
        self,
        key: Expression,
        computed: bool,
        offset: int,
    ) -> JsProperty:
        if self._eat(JsTokenKind.COLON):
            value = self._parse_assignment_expression()
            return JsProperty(
                key=key, value=value, computed=computed,
                shorthand=False, offset=offset)
        if not computed and self._eat(JsTokenKind.EQUALS):
            right = self._parse_assignment_expression()
            value = JsAssignmentPattern(left=key, right=right, offset=key.offset)
            return JsProperty(
                key=key, value=value, computed=computed,
                shorthand=True, offset=offset)
        return JsProperty(
            key=key, value=key, computed=computed,
            shorthand=True, offset=offset)

    def _make_method_property(
        self,
        key: Expression,
        kind: JsPropertyKind,
        is_generator: bool,
        offset: int,
        computed: bool = False,
        is_async: bool = False,
    ) -> JsProperty:
        func_offset = self._current.offset
        with self._function_body_context(is_async, is_generator):
            params = self._parse_formal_parameters()
            body = self._parse_block_statement()
        value = JsFunctionExpression(
            params=params, body=body, generator=is_generator,
            is_async=is_async, offset=func_offset)
        return JsProperty(
            key=key, value=value, computed=computed,
            shorthand=False, method=True, kind=kind, offset=offset)

    def _parse_property_key(self) -> tuple[Expression, bool]:
        if self._at(JsTokenKind.LBRACKET):
            self._advance()
            key = self._parse_assignment_expression()
            self._expect(JsTokenKind.RBRACKET)
            return key, True
        return self._parse_property_name(), False

    def _parse_property_name(self) -> Expression:
        tok = self._current
        if self._at(JsTokenKind.INTEGER, JsTokenKind.FLOAT):
            self._advance()
            raw = tok.value
            text = raw.replace('_', '')
            return JsNumericLiteral(
                value=float(text) if tok.kind == JsTokenKind.FLOAT else self._parse_int_text(text),
                raw=raw,
                offset=tok.offset,
            )
        if self._at(JsTokenKind.STRING_SINGLE, JsTokenKind.STRING_DOUBLE):
            return self._parse_string_literal()
        if self._at(JsTokenKind.PRIVATE_IDENTIFIER):
            self._advance()
            return JsPrivateIdentifier(name=tok.value[1:], offset=tok.offset)
        self._advance()
        return self._name_or_error(tok.value, tok.offset)

    def _parse_paren_or_arrow(self, is_async: bool = False) -> Expression:
        """
        What ECMA-262 calls `CoverParenthesizedExpressionAndArrowParameterList`: a bracketed list
        that the token behind the closing bracket decides the reading of, because nothing inside it
        does. It is read as a list of assignment expressions either way and only then converted,
        which is what lets one pass read a head no expression grammar accepts.

        Three of its shapes belong to the parameter reading alone and are not expressions at all —
        the empty list, a rest element, and a trailing comma — so a list holding one of them is an
        arrow head or it is nothing. The rest element in particular may only stand last, and reading
        it as one of the list rather than as a case of its own is the whole difference between
        `(...a) => a` and `(b, ...a) => a`.

        Where such a list has no arrow behind it, the head stands with nothing to give its
        parameters, and what is missing is recorded where the body would be. Demanding the arrow
        instead consumes whatever does stand there — the semicolon ending the statement, say —
        and the body then reads the statement behind it, so `x = (a,); y = 2;` would take the
        second line into a function nobody wrote and leave nothing to say that it had.
        """
        with self._with_no_in(False):
            offset = self._current.offset
            self._expect(JsTokenKind.LPAREN)

            items: list[Expression] = []
            head_only = True

            while not self._at(JsTokenKind.RPAREN, JsTokenKind.EOF):
                if self._at(JsTokenKind.ELLIPSIS):
                    items.append(self._parse_rest_element())
                    head_only = True
                    break
                items.append(self._parse_assignment_expression())
                head_only = False
                if not self._eat(JsTokenKind.COMMA):
                    break
                head_only = self._at(JsTokenKind.RPAREN)

            self._expect(JsTokenKind.RPAREN)

            if self._at(JsTokenKind.ARROW) and not self._preceded_by_newline:
                self._advance()
                body = self._parse_arrow_body(is_async)
            elif head_only:
                body = JsErrorNode(
                    text='',
                    message='a parameter list with no arrow behind it',
                    offset=self._current.offset,
                )
            else:
                expression = items[0] if len(items) == 1 else JsSequenceExpression(
                    expressions=items, offset=offset)
                return JsParenthesizedExpression(expression=expression, offset=offset)

            return JsArrowFunctionExpression(
                params=[self._to_param(item) for item in items],
                body=body,
                offset=offset,
            )

    def _parse_arrow_body(self, is_async: bool = False) -> Expression | JsBlockStatement:
        with self._function_body_context(is_async, False):
            if self._at(JsTokenKind.LBRACE):
                return self._parse_block_statement()
            return self._parse_assignment_expression()

    def _to_param(self, expr: Expression) -> Expression:
        if isinstance(expr, JsIdentifier):
            return expr
        if isinstance(expr, JsAssignmentExpression) and expr.operator == '=':
            return JsAssignmentPattern(
                left=self._to_param(expr.left),
                right=expr.right,
                offset=expr.offset,
            )
        if isinstance(expr, JsSpreadElement):
            return JsRestElement(argument=self._to_param(expr.argument), offset=expr.offset)
        if isinstance(expr, JsArrayExpression):
            elements = [
                self._to_param(e) if e is not None else None
                for e in expr.elements
            ]
            return JsArrayPattern(elements=elements, offset=expr.offset)
        if isinstance(expr, JsObjectExpression):
            props: list[JsProperty | JsRestElement] = []
            for p in expr.properties:
                if isinstance(p, JsSpreadElement):
                    props.append(JsRestElement(
                        argument=self._to_param(p.argument), offset=p.offset))
                else:
                    props.append(p)
            return JsObjectPattern(properties=props, offset=expr.offset)
        return expr

    def _parse_function_expression(self) -> JsFunctionExpression:
        return self._parse_function_impl(as_expression=True)

    def _parse_class_expression(self) -> JsClassExpression:
        return self._parse_class_impl(as_expression=True)

    def _parse_async_expression(self) -> Expression:
        offset = self._current.offset
        self._advance()
        return self._parse_expression_starting_with_async(offset)

    def _parse_expression_starting_with_async(self, offset: int) -> Expression:
        if not self._preceded_by_newline:
            if self._at(JsTokenKind.FUNCTION):
                return self._parse_function_impl(as_expression=True, is_async=True)

            if self._at(JsTokenKind.ARROW):
                self._advance()
                param = JsIdentifier(name='async', offset=offset)
                body = self._parse_arrow_body(False)
                return JsArrowFunctionExpression(
                    params=[param], body=body, is_async=False, offset=offset)

            if (
                self._at_binding_identifier()
                and self._peek_next().kind == JsTokenKind.ARROW
                and not self._ahead_newline
            ):
                tok = self._advance()
                self._advance()
                param = JsIdentifier(name=tok.value, offset=tok.offset)
                body = self._parse_arrow_body(True)
                return JsArrowFunctionExpression(
                    params=[param], body=body, is_async=True, offset=offset)

            if self._at(JsTokenKind.LPAREN):
                self._advance()
                with self._with_no_in(False):
                    args = self._parse_argument_list()
                if self._at(JsTokenKind.ARROW) and not self._preceded_by_newline:
                    self._advance()
                    params = [self._to_param(arg) for arg in args]
                    body = self._parse_arrow_body(True)
                    return JsArrowFunctionExpression(
                        params=params, body=body, is_async=True, offset=offset)
                return JsCallExpression(
                    callee=JsIdentifier(name='async', offset=offset),
                    arguments=args,
                    optional=False,
                    offset=offset,
                )

        return JsIdentifier(name='async', offset=offset)

    def _parse_yield_expression(self) -> JsYieldExpression:
        """
        A YieldExpression, whose one line terminator restriction sits between the `yield` and what
        follows it. A newline there ends the expression, so neither a `*` nor an argument can still
        belong to it; a newline anywhere after the `*` is ordinary whitespace, and the argument is
        read across it.

        A token that closes the construct the `yield` stands in is not an argument, and the hole of
        a template is closed by the text that resumes it rather than by a brace of its own.
        """
        offset = self._current.offset
        self._advance()
        if self._preceded_by_newline:
            return JsYieldExpression(argument=None, delegate=False, offset=offset)
        delegate = self._eat(JsTokenKind.STAR) is not None
        argument = None
        if not self._at(
            JsTokenKind.SEMICOLON,
            JsTokenKind.RBRACE,
            JsTokenKind.RPAREN,
            JsTokenKind.RBRACKET,
            JsTokenKind.COMMA,
            JsTokenKind.COLON,
            JsTokenKind.TEMPLATE_MIDDLE,
            JsTokenKind.TEMPLATE_TAIL,
            JsTokenKind.EOF,
        ):
            argument = self._parse_assignment_expression()
        return JsYieldExpression(
            argument=argument, delegate=delegate, offset=offset)
