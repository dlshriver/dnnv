import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_GeneratorExp():
    spec_str = "(i for i in range(5))"
    with pytest.raises(
        DNNPParserError,
        match="DNNP does not currently support generator expressions",
    ):
        _ = parse_str(spec_str)
