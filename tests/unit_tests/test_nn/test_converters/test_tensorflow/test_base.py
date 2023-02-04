import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *
from dnnv.utils import get_subclasses


def test_missing():
    class _FakeOperation(Operation):
        pass

    with pytest.raises(ValueError) as excinfo:
        TensorflowConverter().visit(_FakeOperation())
    assert str(excinfo.value).startswith(
        "Tensorflow converter not implemented for operation type"
    )


def test_has_all():
    converter = TensorflowConverter()
    for operation in get_subclasses(Operation):
        if operation.__name__.startswith("_"):
            continue
        assert hasattr(converter, f"visit_{operation}")


# def test_convert():
#     # TODO
