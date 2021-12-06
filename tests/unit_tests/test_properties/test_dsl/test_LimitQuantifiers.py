import pytest

from dnnv.properties import *
from dnnv.properties.dsl import LimitQuantifiers, PropertyParserError


def test_non_top_level():
    x = Symbol("x")
    expr = And(Forall(x, Constant(0) < x), Forall(x, x < Constant(10)))
    with pytest.raises(
        PropertyParserError, match="Quantifiers are only allowed at the top level"
    ):
        _ = LimitQuantifiers()(expr)

    expr = And(Exists(x, Constant(0) < x), Exists(x, x < Constant(10)))
    with pytest.raises(
        PropertyParserError, match="Quantifiers are only allowed at the top level"
    ):
        _ = LimitQuantifiers()(expr)


def test_mixed():
    x = Symbol("x")
    y = Symbol("y")
    expr = Exists(x, Forall(y, x > y))
    with pytest.raises(
        PropertyParserError,
        match="Quantifiers at the top level must be of the same type",
    ):
        _ = LimitQuantifiers()(expr)

    expr = Forall(x, Exists(y, x > y))
    with pytest.raises(
        PropertyParserError,
        match="Quantifiers at the top level must be of the same type",
    ):
        _ = LimitQuantifiers()(expr)
