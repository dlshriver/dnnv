import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_ListComp():
    spec_str = "[i for i in range(5)]"
    with pytest.raises(
        DNNPParserError,
        match="DNNP does not currently support list comprehensions",
    ):
        _ = parse_str(spec_str)
