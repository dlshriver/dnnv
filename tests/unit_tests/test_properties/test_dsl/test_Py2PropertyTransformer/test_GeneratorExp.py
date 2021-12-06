import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_GeneratorExp():
    node = ast.parse("(i for i in range(5))")
    with pytest.raises(
        PropertyParserError,
        match="DNNP does not currently support generator expressions",
    ):
        _ = parse_ast(node)
