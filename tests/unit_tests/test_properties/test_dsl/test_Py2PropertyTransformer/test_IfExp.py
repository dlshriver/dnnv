import ast

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast


def test_IfExp():
    node = ast.parse("T if x else F")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(IfThenElse(Symbol("x"), Symbol("T"), Symbol("F")))
