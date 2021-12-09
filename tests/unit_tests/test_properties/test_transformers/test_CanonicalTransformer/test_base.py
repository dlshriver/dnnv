import pytest

from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer
from dnnv.utils import get_subclasses


@pytest.mark.xfail
def test_has_all_visitors():
    transformer = CanonicalTransformer()
    for expr_t in get_subclasses(Expression):
        if expr_t in (
            ArithmeticExpression,
            AssociativeExpression,
            BinaryExpression,
            LogicalExpression,
            TernaryExpression,
            Quantifier,
            UnaryExpression,
        ):
            continue
        expr_t_name = expr_t.__name__
        if expr_t_name == "FakeExpression":
            continue
        assert hasattr(
            transformer, f"visit_{expr_t_name}"
        ), f"Missing visitor for {expr_t}"
