import numpy as np

from dnnv.verifiers.common.reductions.iopolytope import *
from dnnv.verifiers.common.reductions.iopolytope import Variable


def setup_function():
    Variable._count = 0


def test_init():
    v = Variable((1, 3, 4, 4))
    assert v.shape == (1, 3, 4, 4)
    assert v.name == "x_0"
    assert Variable._count == 1

    v = Variable((1, 5))
    assert v.shape == (1, 5)
    assert v.name == "x_1"
    assert Variable._count == 2

    v = Variable((1, 10), "test_var")
    assert v.shape == (1, 10)
    assert v.name == "test_var"
    assert Variable._count == 3


def test_size():
    v = Variable((1, 3, 4, 4), "test_var")
    assert v.size() == 48


def test_str():
    v = Variable((1, 10))
    assert str(v) == "x_0"

    v = Variable((1, 3, 4, 4), "test_var")
    assert str(v) == "test_var"


def test_repr():
    v = Variable((1, 10))
    assert repr(v) == "Variable((1, 10), 'x_0')"

    v = Variable((1, 3, 4, 4), "test_var")
    assert repr(v) == "Variable((1, 3, 4, 4), 'test_var')"


def test_hash():
    sizes = np.random.randint(
        np.ones(100, dtype=int), np.full(100, 25, dtype=int)
    )
    hashes = set()
    for size in sizes:
        v = Variable((1, size))
        hashes.add(hash(v))
    assert len(hashes) == 100

    assert hash(Variable((1, sizes[0]), "x_0")) in hashes


def test_eq():
    v1 = Variable((1, 100))
    assert v1 != 99

    v2 = Variable((1, 100))
    assert v1 != v2

    v3 = Variable((1, 100), "x_0")
    assert v1 == v3
