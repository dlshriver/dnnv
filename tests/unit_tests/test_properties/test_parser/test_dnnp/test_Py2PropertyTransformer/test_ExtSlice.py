import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_ExtSlice_default_slices():
    spec_str = "x[:,:]"
    phi = parse_str(spec_str)
    with phi.ctx:
        x = Symbol("x")
        assert phi.is_equivalent(x[Constant(slice(None)), Constant(slice(None))])


def test_ExtSlice_slice_int():
    spec_str = "x[:,0]"
    phi = parse_str(spec_str)
    with phi.ctx:
        x = Symbol("x")
        assert phi.is_equivalent(x[Constant(slice(None)), Constant(0)])


def test_ExtSlice_non_primitive():
    spec_str = "x[:, i, j, k]"
    with pytest.raises(
        DNNPParserError,
        match="DNNP does not currently support definition of (.)*? containing non-primitive types",
    ):
        _ = parse_str(spec_str)
