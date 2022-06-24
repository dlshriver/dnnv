import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_Await():
    spec_str = "await f()"
    with pytest.raises(
        DNNPParserError, match="DNNP does not support await expressions"
    ):
        _ = parse_str(spec_str)
