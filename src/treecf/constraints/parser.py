"""Hand-rolled recursive-descent parser for string constraint sugar.

Grammar (EBNF):
    constraint := expr op expr
    op         := "<=" | ">=" | "=="
    expr       := term (("+"|"-") term)*
    term       := [number "*"] feature | number
    feature    := identifier

Only linear expressions are expressible by design; anything richer must be
written as constraint objects. Errors carry a caret marking the offending token.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from treecf._errors import ConstraintParseError
from treecf.constraints.objects import Linear

_TOKEN = re.compile(
    r"\s*(?:(?P<op><=|>=|==)|(?P<num>\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<sym>[+\-*])|(?P<bad>\S))"
)


class _Token:
    def __init__(self, kind: str, text: str, pos: int) -> None:
        self.kind = kind
        self.text = text
        self.pos = pos


def constraint(text: str, feature_names: Sequence[str] | None = None) -> Linear:
    """Parse ``"2*a - b <= 3"``-style sugar into a canonical Linear object.

    When ``feature_names`` is given, identifiers are validated immediately;
    otherwise validation happens later in ``compile_constraints``.
    """
    tokens = _tokenize(text)
    parser = _Parser(text, tokens, feature_names)
    return parser.parse()


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    for match in _TOKEN.finditer(text):
        kind = str(match.lastgroup)
        value = str(match.group(kind))
        if kind == "bad":
            raise ConstraintParseError(_caret(text, match.start(kind), f"unexpected {value!r}"))
        tokens.append(_Token(kind, value, match.start(kind)))
    return tokens


class _Parser:
    def __init__(
        self, text: str, tokens: list[_Token], feature_names: Sequence[str] | None
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.names = set(feature_names) if feature_names is not None else None
        self.i = 0

    def parse(self) -> Linear:
        lhs_coeffs, lhs_const = self._expr()
        op_token = self._peek()
        if op_token is None or op_token.kind != "op":
            pos = op_token.pos if op_token else len(self.text)
            raise ConstraintParseError(_caret(self.text, pos, "expected an operator <=, >=, =="))
        self.i += 1
        rhs_coeffs, rhs_const = self._expr()
        trailing = self._peek()
        if trailing is not None:
            raise ConstraintParseError(
                _caret(self.text, trailing.pos, f"unexpected trailing {trailing.text!r}")
            )

        coefficients: dict[str, float] = dict(lhs_coeffs)
        for name, coef in rhs_coeffs.items():
            coefficients[name] = coefficients.get(name, 0.0) - coef
        coefficients = {n: c for n, c in coefficients.items() if c != 0.0}
        if not coefficients:
            raise ConstraintParseError(
                _caret(self.text, 0, "constraint references no feature")
            )
        return Linear(coefficients=coefficients, op=op_token.text, rhs=rhs_const - lhs_const)

    def _expr(self) -> tuple[dict[str, float], float]:
        coefficients: dict[str, float] = {}
        constant = 0.0
        sign = 1.0
        first = True
        while True:
            token = self._peek()
            if token is None or token.kind == "op":
                if first:
                    pos = token.pos if token else len(self.text)
                    raise ConstraintParseError(_caret(self.text, pos, "expected an expression"))
                return coefficients, constant
            if token.kind == "sym" and token.text in "+-":
                if first:
                    sign = -1.0 if token.text == "-" else 1.0
                    self.i += 1
                    self._term(coefficients, sign)
                else:
                    self.i += 1
                    self._term(coefficients, -1.0 if token.text == "-" else 1.0)
            elif first:
                self._term(coefficients, 1.0)
            else:
                return coefficients, constant
            # fold constants accumulated by _term via sentinel key
            constant += coefficients.pop("", 0.0)
            first = False

    def _term(self, coefficients: dict[str, float], sign: float) -> None:
        token = self._peek()
        if token is None:
            raise ConstraintParseError(_caret(self.text, len(self.text), "expected a term"))
        if token.kind == "num":
            self.i += 1
            number = float(token.text)
            nxt = self._peek()
            if nxt is not None and nxt.kind == "sym" and nxt.text == "*":
                self.i += 1
                name_token = self._peek()
                if name_token is None or name_token.kind != "name":
                    pos = name_token.pos if name_token else len(self.text)
                    raise ConstraintParseError(_caret(self.text, pos, "expected a feature after *"))
                self.i += 1
                self._add_feature(coefficients, name_token, sign * number)
            else:
                coefficients[""] = coefficients.get("", 0.0) + sign * number
        elif token.kind == "name":
            self.i += 1
            self._add_feature(coefficients, token, sign)
        else:
            raise ConstraintParseError(
                _caret(self.text, token.pos, f"expected a term, found {token.text!r}")
            )

    def _add_feature(self, coefficients: dict[str, float], token: _Token, coef: float) -> None:
        if self.names is not None and token.text not in self.names:
            raise ConstraintParseError(
                _caret(self.text, token.pos, f"unknown feature {token.text!r}")
            )
        coefficients[token.text] = coefficients.get(token.text, 0.0) + coef

    def _peek(self) -> _Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None


def _caret(text: str, pos: int, message: str) -> str:
    return f"{message}\n  {text}\n  {' ' * pos}^"
