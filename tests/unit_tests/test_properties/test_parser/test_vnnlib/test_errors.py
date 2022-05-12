import re

import pytest

from dnnv.properties.expressions import *
from dnnv.properties.parser.vnnlib import VNNLIBParserError, parse_str


def test_bad_token_parse():
    with pytest.raises(
        VNNLIBParserError, match=re.escape("line 1, column 2: unexpected token '+'")
    ):
        _ = parse_str("(+ 3 2 1)")


def test_bad_char_parse():
    with pytest.raises(
        VNNLIBParserError, match="line 1, column 1: unexpected character '\"'"
    ):
        _ = parse_str('"')


def test_multi_arg_subtract():
    with pytest.raises(
        VNNLIBParserError, match="Subtraction not implemented for more than 2 args."
    ):
        _ = parse_str("(assert (= (- 3 2 1) 0))")


def test_declare_twice():
    with pytest.raises(
        VNNLIBParserError, match="line 2: identifier 'varname' already exists"
    ):
        _ = parse_str("(declare-const varname Real)\n(declare-const varname Real)")


def test_unknown_identifier():
    with pytest.raises(
        VNNLIBParserError, match="line 1: unknown identifier 'badidentifier'"
    ):
        _ = parse_str("(assert (badidentifier 4))")


def test_negated_identifier():
    with pytest.raises(VNNLIBParserError, match="line 2: unknown identifier '-x'"):
        _ = parse_str("(declare-const x Real)\n(assert (> -x 0))")


def test_unknown_sort():
    with pytest.raises(
        NotImplementedError, match="line 1: sort badsort is not currently supported"
    ):
        _ = parse_str("(declare-const X badsort)")


def test_let():
    with pytest.raises(
        NotImplementedError, match="line 2: 'let' is not currently supported"
    ):
        _ = parse_str("(declare-const x Real)\n(assert (let ((a x)) (> a 0)))")


def test_parameterized_sorts():
    with pytest.raises(
        NotImplementedError,
        match="line 1: parameterized sorts are not currently supported",
    ):
        _ = parse_str("(declare-const A (Set Real))")


def test_qual_identifiers():
    with pytest.raises(
        NotImplementedError,
        match="line 1: qualified identifiers are not currently supported",
    ):
        _ = parse_str("(assert (as nil Real))")
