import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_DictComp():
    spec_str = r"{i: 0 for i in range(5)}"
    with pytest.raises(
        DNNPParserError,
        match="DNNP does not currently support dict comprehensions",
    ):
        _ = parse_str(spec_str)
