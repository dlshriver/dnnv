import pytest

from dnnv.properties.base import *
from dnnv.properties.transformers import ToCNF


def test_missing():
    class FakeExpression(Expression):
        pass

    with pytest.raises(
        ValueError, match="Unimplemented expression type: FakeExpression"
    ):
        ToCNF().visit(FakeExpression())
    del FakeExpression
