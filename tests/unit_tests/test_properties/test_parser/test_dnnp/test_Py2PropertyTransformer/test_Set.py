import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_Set_non_empty():
    spec_str = "{0, 1, 2, 3, 4}"
    phi = parse_str(spec_str)
    assert phi == {0, 1, 2, 3, 4}


def test_Set_non_primitive():
    spec_str = "{'a', x}"
    with pytest.raises(
        DNNPParserError,
        match="DNNP does not currently support definition of sets containing non-primitive types",
    ):
        _ = parse_str(spec_str)
