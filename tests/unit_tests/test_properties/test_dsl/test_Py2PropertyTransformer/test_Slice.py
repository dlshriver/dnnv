import ast

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast


def test_Slice_empty():
    node = ast.parse("x[:]")
    phi = parse_ast(node)
    with phi.ctx:
        x = Symbol("x")
        assert phi.is_equivalent(x[Constant(slice(None))])


def test_Slice_start_end():
    node = ast.parse("x[b:e]")
    phi = parse_ast(node)
    with phi.ctx:
        x = Symbol("x")
        b = Symbol("b")
        e = Symbol("e")
        assert phi.is_equivalent(x[Slice(b, e, Constant(None))])


def test_Slice_start_end_step():
    node = ast.parse("x[b:e:s]")
    phi = parse_ast(node)
    with phi.ctx:
        x = Symbol("x")
        b = Symbol("b")
        e = Symbol("e")
        s = Symbol("s")
        assert phi.is_equivalent(x[Slice(b, e, s)])
