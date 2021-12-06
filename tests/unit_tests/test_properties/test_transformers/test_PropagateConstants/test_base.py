import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants
from dnnv.utils import get_subclasses


def test_missing():
    class FakeExpression(Expression):
        pass

    with pytest.raises(
        ValueError, match="Unimplemented expression type: FakeExpression"
    ):
        PropagateConstants().visit(FakeExpression())
    del FakeExpression


@pytest.mark.xfail
def test_has_all_visitors():
    transformer = PropagateConstants()
    for expr_t in get_subclasses(Expression):
        if expr_t in (
            AssociativeExpression,
            BinaryExpression,
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
