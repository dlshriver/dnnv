import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError, Py2PropertyTransformer


def test_Call_non_name():
    node = ast.parse("f()()")
    phi = parse_ast(node)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("f")()())
