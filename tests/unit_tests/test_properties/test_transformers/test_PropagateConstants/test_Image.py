import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Image(tmp_path):
    transformer = PropagateConstants()

    arr = np.random.randn(3, 4, 4)
    np.save(tmp_path / "test.npy", arr)
    expr = Image(Constant(tmp_path / "test.npy"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, arr)
