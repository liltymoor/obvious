from __future__ import annotations

import enum
from functools import partial
from io import SEEK_SET, TextIOBase
from typing import Callable


def parse_keyword(keyword: str, text: TextIOBase) -> str:
    s = text.read(len(keyword))
    if s == keyword:
        return keyword
    return ""


def parse_num(text: TextIOBase) -> str:
    res = ""
    cur = text.read(1)
    if cur.isdecimal():
        res += cur
        pos = text.tell()
        cur = text.read(1)
        while cur.isdecimal():
            res += cur
            pos = text.tell()
            cur = text.read(1)
        else:
            text.seek(pos, SEEK_SET)
    return res


def parse_string(text: TextIOBase) -> str:
    res = ""
    start = text.read(1)
    if start == '"':
        cur = text.read(1)
        while cur and cur != '"':
            res += cur
            cur = text.read(1)
        if cur != '"':
            return ""
    return res

def parse_var(text: TextIOBase) -> str:
    res = ""
    cur = text.read(1)
    if cur.isalpha() or cur == "_":
        res += cur
        pos = text.tell()
        cur = text.read(1)
        while cur.isalnum() or cur == "_":
            res += cur
            pos = text.tell()
            cur = text.read(1)
        else:
            text.seek(pos, SEEK_SET)
    return res


@enum.unique
class TokenType(enum.Enum):
    IF = ("if",)
    WHILE = ("while",)

    ECHO = ("echo",)
    STRCAT = ("strcat",)

    INTERRUPT = ("interrupt",)

    L_BRACE = ("{",)
    R_BRACE = ("}",)
    L_PAR = ("(",)
    R_PAR = (")",)
    L_BRACKET = ("[",)
    R_BRACKET = ("]",)
    SEMICOLON = (";",)
    COMMA = (",",)

    NEQ = ("!=",)
    EQ = ("==",)
    NOT = ("!",)
    LT = ("<",)
    GT = (">",)
    AND = ("&",)
    OR = ("|",)

    ASSIGN = ("=",)
    PLUS = ("+",)
    MINUS = ("-",)
    MUL = ("*",)
    DIV = ("/",)
    MOD = ("%",)

    NUM_LIT = (None, parse_num)
    STR_LIT = (None, parse_string)
    VAR_LIT = (None, parse_var)

    EOF = ("",)

    def __init__(self, keyword: str | None, rule: Callable[[TextIOBase], str] | None = None):
        if keyword is None:
            assert rule is not None
            self._rule = rule
        else:
            self._keyword = keyword
            self._rule = partial(parse_keyword, keyword)


class Token:
    def __init__(self, token_type: TokenType, value: str, line: int) -> None:
        self.token_type = token_type
        self.value = value
        self.line = line

    def __str__(self) -> str:
        return self.token_type.name


class Tokenizer:
    def __init__(self, text: TextIOBase) -> None:
        self.text = text
        self.stack: list[Token] = []
        self.line = 0
    def get_next_token(self) -> Token | None:
        if self.stack:
            return self.stack.pop()

        # seek for default tokens
        for token_type in TokenType:
            pos = self.text.tell()
            val = token_type._rule(self.text)
            if val != "":
                return Token(token_type, val, self.line)
            self.text.seek(pos, SEEK_SET)

        pos = self.text.tell()
        cur = self.text.read(1)
        if cur.isspace():
            if cur == "\n":
                self.line += 1
            return self.get_next_token()
        if cur == "":
            return Token(TokenType.EOF, "", self.line)
        self.text.seek(pos, SEEK_SET)
        return None
