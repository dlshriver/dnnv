import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *
from dnnv.utils import get_subclasses


def test_missing():
    class FakeOperation(Operation):
        pass

    with pytest.raises(ValueError) as excinfo:
        TensorflowConverter().visit(FakeOperation())
    assert str(excinfo.value).startswith(
        "TensorflowConverter not implemented for operation type"
    )
    del FakeOperation


def test_has_all():
    converter = TensorflowConverter()
    for operation in get_subclasses(Operation):
        if "Fake" in repr(operation):
            continue
        assert hasattr(converter, f"visit_{operation}")


# def test_convert():
#     # TODO
