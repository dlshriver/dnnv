import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_Yield():
    spec_str = "yield ..."
    with pytest.raises(
        DNNPParserError, match="DNNP does not support yield expressions"
    ):
        _ = parse_str(spec_str)
