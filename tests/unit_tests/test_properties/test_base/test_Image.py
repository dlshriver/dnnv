import numpy as np
import pytest

from dnnv.properties.base import *


def test_load(tmp_path):
    path = tmp_path / "test.npy"
    img_arr = np.random.randn(3, 32, 32).astype(np.float32)
    np.save(path, img_arr)

    img = Image.load(Constant(path))
    assert isinstance(img, Constant)
    img_val = img.value
    assert img_val.shape == (1, 3, 32, 32)
    assert np.allclose(img_val, img_arr[None])

    img = Image.load(Symbol("x"))
    assert isinstance(img, Image)
    assert img.path == Symbol("x")


def test_value(tmp_path):
    path = tmp_path / "test.npy"
    img_arr = np.random.randn(3, 32, 32).astype(np.float32)
    np.save(path, img_arr)

    img = Image(Constant(path))
    img_val = img.value
    assert img_val.shape == (1, 3, 32, 32)
    assert np.allclose(img_val, img_arr[None])

    img = Image(Symbol("x"))
    with pytest.raises(ValueError):
        img_val = img.value


def test_repr(tmp_path):
    img = Image(Constant(str(tmp_path)))
    assert repr(img) == f"Image('{str(tmp_path)}')"

    img = Image(Symbol("x"))
    assert repr(img) == "Image(Symbol('x'))"


def test_str(tmp_path):
    img = Image(Constant(str(tmp_path)))
    assert str(img) == f"Image({str(tmp_path)})"

    img = Image(Symbol("x"))
    assert str(img) == "Image(x)"
