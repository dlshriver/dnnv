import ast
import astunparse
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError, Py2PropertyTransformer


def test_Compare_single():
    node = ast.parse("x < 0")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") < Constant(0))

    node = ast.parse("x <= 0")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") <= Constant(0))

    node = ast.parse("x > 0")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") > Constant(0))

    node = ast.parse("x >= 0")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") >= Constant(0))

    node = ast.parse("x == 0")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") == Constant(0))

    node = ast.parse("x != 0")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") != Constant(0))


def test_Compare_multi():
    node = ast.parse("0 < x < 1")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Constant(0) < Symbol("x"), Symbol("x") < Constant(1))
        )

    node = ast.parse("0 < x <= 1")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Constant(0) < Symbol("x"), Symbol("x") <= Constant(1))
        )

    node = ast.parse("a > b > c")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Symbol("a") > Symbol("b"), Symbol("b") > Symbol("c"))
        )

    node = ast.parse("a >= 0 >= b")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Symbol("a") >= Constant(0), Constant(0) >= Symbol("b"))
        )

    node = ast.parse("a == b != c")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Symbol("a") == Symbol("b"), Symbol("b") != Symbol("c"))
        )


def test_Compare_unsupported():
    node = ast.parse("a in b in c")
    with pytest.raises(PropertyParserError, match="Unsupported comparison function:"):
        _ = parse_ast(node)
