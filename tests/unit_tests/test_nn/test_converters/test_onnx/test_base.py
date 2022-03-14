import pytest

from dnnv.nn import OperationGraph
from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *
from dnnv.utils import get_subclasses


def test_missing():
    class FakeOperation(Operation):
        pass

    with pytest.raises(ValueError) as excinfo:
        OnnxConverter().visit(FakeOperation())
    assert str(excinfo.value).startswith(
        "OnnxConverter not implemented for operation type"
    )
    del FakeOperation


def test_has_all():
    converter = OnnxConverter()
    for operation in get_subclasses(Operation):
        if "Fake" in repr(operation):
            continue
        assert hasattr(converter, f"visit_{operation}")


# def test_convert():
#     # TODO
