import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_Starred():
    spec_str = "f(*[1, 2, 3])"
    with pytest.raises(
        DNNPParserError,
        match="DNNP does not currently support starred expressions",
    ):
        _ = parse_str(spec_str)
