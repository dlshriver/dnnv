import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference, DNNVShapeError, DNNVTypeError


def test_numeric():
    inference = DetailsInference()

    c = Constant(1003)
    inference.visit(c)
    assert inference.shapes[c].value == ()
    assert inference.types[c].value == np.min_scalar_type(1003)

    c = Constant(3.141592)
    inference.visit(c)
    assert inference.shapes[c].value == ()
    assert inference.types[c].value == np.min_scalar_type(3.141592)


def test_sequence():
    inference = DetailsInference()

    c = Constant([1, 2, 3])
    inference.visit(c)
    assert inference.shapes[c].value == (3,)
    assert inference.types[c].value == np.asarray([1, 2, 3]).dtype

    c = Constant(((1.1, 4, 10066.794),))
    inference.visit(c)
    assert inference.shapes[c].value == (1, 3)
    assert inference.types[c].value == np.asarray(((1.1, 4, 10066.794),)).dtype


def test_array():
    inference = DetailsInference()

    c = Constant(np.random.randn(3, 5, 8))
    inference.visit(c)
    assert inference.shapes[c].value == (3, 5, 8)
    assert inference.types[c].value == np.float64


def test_objects():
    inference = DetailsInference()

    class Test:
        pass

    c = Constant(Test())
    inference.visit(c)
    assert inference.shapes[c].value == ()
    assert inference.types[c].value == Test

    c = Constant(Test)
    inference.visit(c)
    assert inference.shapes[c].value == ()
    assert inference.types[c].value == type

    c = Constant(int)
    inference.visit(c)
    assert inference.shapes[c].value == ()
    assert inference.types[c].value == type(int)
