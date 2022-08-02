from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import lark
import numpy as np

from dnnv.properties.expressions.terms import symbol

from ...expressions import *
from ..utils import LimitQuantifiers, parse_cli
from .errors import VNNLIBParserError

# TODO : implement full smt-lib grammar
# currently enough for vnn-lib spec?
# partial smt-lib 2.6 grammar
grammar = """//partial smt-lib 2.6 grammar
script          : command*
command         : "(" "assert" term ")"                 -> cmd_assert
                | "(" "declare-const" SYMBOL sort ")"   -> cmd_declare_const
?term           : spec_constant
                | qual_identifier
                | "(" qual_identifier term+ ")"
                | "(" "let" "(" var_binding+ ")" term ")" -> term_let
?spec_constant  : DECIMAL
                | NUMERAL
                | HEXADECIMAL
                | BINARY
                | STRING
?qual_identifier: identifier
                | "(" "as" identifier sort ")"
var_binding     : "(" SYMBOL term ")"
sort            : identifier
                | "(" identifier sort+ ")"              -> parameterized_sort
?identifier     : SYMBOL
                | "(" "_" SYMBOL index+ ")"
index           : NUMERAL
                | SYMBOL

// terminals
SYMBOL          : (LETTER|SYMBOL_CHAR) (DIGIT|LETTER|SYMBOL_CHAR)*
SYMBOL_CHAR     : /[~!@$%^&*+=<>.?\/_-]/
BINARY          : /"#b"[01]+/
HEXADECIMAL     : /"#x"[0-9A-Fa-f]+/
DECIMAL         : NUMERAL "." "0"* NUMERAL?
NUMERAL         : /[1-9][0-9]*/
                | "0"
STRING          : "\\"" (PRINTABLE_CHAR|/[\\u0009\\u000A\\u000D\\u0020]/)* "\\""
PRINTABLE_CHAR  : /[\\u0020-\\u0021]/
                | /[\\u0023-\\u007E]/
                | /[\\u0080-\\uFFFF]/
                | "\\"\\""
COMMENT         : /;[\\t -~]*[\\r\\n]/

%import common.DIGIT
%import common.LETTER
%import common.WS
%ignore WS
%ignore COMMENT
"""


def subtraction_helper(*args):
    if len(args) == 1:
        return Negation(*args)
    elif len(args) == 2:
        return Subtract(*args)
    raise VNNLIBParserError("Subtraction not implemented for more than 2 args.")


class VNNLIBTransformer(lark.Transformer):
    _OPERATORS: Dict[str, Callable] = {
        "+": Add,
        "/": Divide,
        ">": GreaterThan,
        ">=": GreaterThanOrEqual,
        "<": LessThan,
        "<=": LessThanOrEqual,
        "=": Equal,
        "*": Multiply,
        "-": subtraction_helper,
        "or": Or,
        "and": And,
        "ite": IfThenElse,
        "not": Not,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._symbols = {}
        self._assertions: Set[Expression] = set()

    DECIMAL = lambda self, x: Constant(float(x))
    NUMERAL = lambda self, x: Constant(int(x))
    STRING = lambda self, x: Constant(str(x))

    @lark.v_args(inline=True)
    def cmd_assert(self, expression):
        return expression

    @lark.v_args(inline=True)
    def cmd_declare_const(self, identifier, sort):
        if identifier in self._symbols:
            raise VNNLIBParserError(
                f"line {identifier.line}: identifier '{identifier}' already exists"
            )
        if identifier.startswith("X_"):
            index = tuple(int(i) for i in identifier.split("_")[1:])
            if len(index) == 1:
                self._symbols[identifier] = Symbol("X")[
                    Call(
                        Constant(np.unravel_index),
                        (Constant(index[0]), Network("N").input_shape[0]),
                        {},
                    )
                ]
            else:
                self._symbols[identifier] = Symbol("X")[index]
        elif identifier.startswith("Y_"):
            index = tuple(int(i) for i in identifier.split("_")[1:])
            if len(index) == 1:
                self._symbols[identifier] = Network("N")(Symbol("X"))[
                    Call(
                        Constant(np.unravel_index),
                        (Constant(index[0]), Network("N").output_shape[0]),
                        {},
                    )
                ]
            else:
                self._symbols[identifier] = Network("N")(Symbol("X"))[index]
        else:
            self._symbols[identifier] = Parameter(identifier, type=sort)
        return lark.Discard

    @lark.v_args(tree=True)
    def parameterized_sort(self, tree):
        raise NotImplementedError(
            f"line {tree.meta.line}: parameterized sorts are not currently supported"
        )

    @lark.v_args(inline=True, meta=True)
    def qual_identifier(self, meta, *children):
        raise NotImplementedError(
            f"line {meta.line}: qualified identifiers are not currently supported"
        )

    def script(self, expressions):
        return ~Exists(Symbol("X"), And(*expressions))

    @lark.v_args(inline=True)
    def sort(self, child):
        if child == "Real":
            return float
        elif child == "Int":
            return int
        elif child == "Bool":
            return bool
        raise NotImplementedError(
            f"line {child.line}: sort {child} is not currently supported"
        )

    def lookup_symbol(self, symbol):
        if symbol in self._symbols:
            return self._symbols[symbol]
        if symbol in self._OPERATORS:
            return self._OPERATORS[symbol]
        if symbol.startswith("-"):
            try:
                return Constant(int(symbol))
            except ValueError:
                pass
            try:
                return Constant(float(symbol))
            except ValueError:
                pass
        raise VNNLIBParserError(f"line {symbol.line}: unknown identifier '{symbol}'")

    @lark.v_args(inline=True, meta=True)
    def term(self, meta, func_symbol, *terms):
        func = self.lookup_symbol(func_symbol)
        args = []
        for term in terms:
            if isinstance(term, lark.Token) and term.type == "SYMBOL":
                args.append(self.lookup_symbol(term))
            else:
                args.append(term)
        return func(*args)

    @lark.v_args(meta=True)
    def term_let(self, meta, children):
        *var_bindings, term = children
        for var_binding in var_bindings:
            del self._symbols[var_binding]
        raise NotImplementedError(f"line {meta.line}: 'let' is not currently supported")

    @lark.v_args(inline=True)
    def var_binding(self, symbol, term):
        self._symbols[symbol] = term
        return symbol


VNNLIBParser = lark.Lark(
    grammar,
    start="script",
    parser="lalr",
    propagate_positions=True,
)


def parse_str(
    spec_str: str, *, path: Path = Path(), args: Optional[List[str]] = None
) -> Expression:
    try:
        parse_tree = VNNLIBParser.parse(spec_str)
    except lark.exceptions.UnexpectedCharacters as unexpected_char_exc:
        raise VNNLIBParserError(
            f"unexpected character '{unexpected_char_exc.char}'",
            lineno=unexpected_char_exc.line,
            col_offset=unexpected_char_exc.column,
        )
    except lark.exceptions.UnexpectedToken as unexpected_token_exc:
        raise VNNLIBParserError(
            f"unexpected token '{unexpected_token_exc.token}'",
            lineno=unexpected_token_exc.line,
            col_offset=unexpected_token_exc.column,
        )
    with Context():
        try:
            phi = VNNLIBTransformer().transform(parse_tree)
        except lark.exceptions.VisitError as e:
            raise e.orig_exc
        LimitQuantifiers()(phi)

        phi = phi.propagate_constants()
        phi = parse_cli(phi, args)
    return phi


def parse(path: Path, args: Optional[List[str]] = None) -> Expression:
    with open(path, "r") as f:
        script_str = f.read()
    return parse_str(script_str, args=args)


__all__ = ["parse", "parse_str"]
