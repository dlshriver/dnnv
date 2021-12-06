import numpy as np
import pytest

from dnnv.properties.expressions.terms.utils import get_callable_name


def test_lambda():
    f = lambda x: x + 1
    name = get_callable_name(f)
    assert name.startswith("test_get_callable_name.<lambda id=0x")


def test_builtin():
    name = get_callable_name(abs)
    assert name == "abs"


def test_function():
    def f(x):
        return x + 1

    name = get_callable_name(f)
    assert name == "test_get_callable_name.test_function.<locals>.f"


def test_method():
    class C:
        def f(self, x):
            return x + 1

    # Not a method, this should take the FunctionType path
    name = get_callable_name(C.f)
    assert name == "test_get_callable_name.test_method.<locals>.C.f"

    name = get_callable_name(C().f)
    assert name == "test_get_callable_name.test_method.<locals>.C.f"

    c = C()
    c.__module__ = "__main__"
    name = get_callable_name(c.f)
    assert name == "__main__.test_method.<locals>.C.f"


def test_np_ufunc():
    name = get_callable_name(np.argmax)
    assert name == "numpy.argmax"


def test_type():
    name = get_callable_name(int)
    assert name == "builtins.int"

    class C:
        pass

    name = get_callable_name(C)
    assert name == "test_get_callable_name.test_type.<locals>.C"


def test_unsupported():
    with pytest.raises(ValueError):
        _ = get_callable_name(exit)
    with pytest.raises(ValueError):
        _ = get_callable_name(help)
