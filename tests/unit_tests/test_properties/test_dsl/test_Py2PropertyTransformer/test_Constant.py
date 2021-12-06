import ast

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast


def test_Ellipsis():
    node = ast.parse("...")
    phi = parse_ast(node)
    assert phi == ...


def test_NameConstant():
    node = ast.parse("None")
    phi = parse_ast(node)
    assert phi == None


def test_Num():
    node = ast.parse("1000")
    phi = parse_ast(node)
    assert phi == 1000


def test_Str():
    node = ast.parse('"test"')
    phi = parse_ast(node)
    assert phi == "test"
