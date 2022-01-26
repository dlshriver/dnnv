import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Image_symbolic():
    inference = DetailsInference()

    expr = Image(Symbol("path"))
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert not inference.types[expr].is_concrete


def test_Image_concrete(tmp_path):
    inference = DetailsInference()

    arr = np.random.rand(3, 32, 32)
    np.save(tmp_path / "test.npy", arr)

    expr = Image(Constant(tmp_path / "test.npy"))
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == arr.shape
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == arr.dtype
