import numpy as np
import pytest

from dnnv.properties.base import _get_function_name


def test_lambda():
    f = lambda x: x + 1
    name = _get_function_name(f)
    assert name.startswith("test__get_function_name.<lambda id=0x")


def test_builtin():
    name = _get_function_name(abs)
    assert name == "abs"


def test_function():
    def f(x):
        return x + 1

    name = _get_function_name(f)
    assert name == "test__get_function_name.test_function.<locals>.f"


def test_method():
    class C:
        def f(self, x):
            return x + 1

    # Not a method, this should take the FunctionType path
    name = _get_function_name(C.f)
    assert name == "test__get_function_name.test_method.<locals>.C.f"

    name = _get_function_name(C().f)
    assert name == "test__get_function_name.test_method.<locals>.C.f"

    c = C()
    c.__module__ = "__main__"
    name = _get_function_name(c.f)
    assert name == "__main__.test_method.<locals>.C.f"


def test_np_ufunc():
    name = _get_function_name(np.argmax)
    assert name == "numpy.argmax"


def test_type():
    name = _get_function_name(int)
    assert name == "builtins.int"

    class C:
        pass

    name = _get_function_name(C)
    assert name == "test__get_function_name.test_type.<locals>.C"


def test_unsupported():
    with pytest.raises(ValueError):
        _ = _get_function_name(exit)
    with pytest.raises(ValueError):
        _ = _get_function_name(help)
