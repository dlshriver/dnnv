import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_YieldFrom():
    spec_str = "yield from ..."
    with pytest.raises(
        DNNPParserError, match="DNNP does not support yield from expressions"
    ):
        _ = parse_str(spec_str)
