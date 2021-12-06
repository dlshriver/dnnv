import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_Assign_name():
    node = ast.parse(
        """from dnnv.properties import *
N = Network("N")
N
    """
    )
    phi = parse_ast(node)
    with phi.ctx:
        assert phi is Network("N")


def test_Assign_non_name():
    node = ast.parse(
        """from dnnv.properties import *
networks = [None, None]
networks[0] = Network("N")
networks[0]
    """
    )
    with pytest.raises(
        PropertyParserError,
        match="Assigning to non-identifiers is not currently supported",
    ):
        _ = parse_ast(node)


def test_Assign_lambda():
    node = ast.parse(
        """from dnnv.properties import *
f = lambda x: x + 1
Forall(x, x < f(x))
    """
    )

    phi = parse_ast(node)
    with phi.ctx:
        x = Symbol("x")
        assert phi.is_equivalent(Forall(x, x < (x + 1)))
