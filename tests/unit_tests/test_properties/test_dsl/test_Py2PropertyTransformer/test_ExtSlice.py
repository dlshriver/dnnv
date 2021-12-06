import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_ExtSlice_default_slices():
    node = ast.parse("x[:,:]")
    phi = parse_ast(node)
    with phi.ctx:
        x = Symbol("x")
        assert phi.is_equivalent(x[Constant(slice(None)), Constant(slice(None))])


def test_ExtSlice_slice_int():
    node = ast.parse("x[:,0]")
    phi = parse_ast(node)
    with phi.ctx:
        x = Symbol("x")
        assert phi.is_equivalent(x[Constant(slice(None)), Constant(0)])


def test_ExtSlice_non_primitive():
    node = ast.parse("x[:, i, j, k]")
    with pytest.raises(
        PropertyParserError,
        match="DNNP does not currently support definition of (.)*? containing non-primitive types",
    ):
        _ = parse_ast(node)
