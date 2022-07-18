import pytest

from dnnv.properties.expressions import Constant
from dnnv.verifiers.common.reductions.base import Property, Reduction


def test_instantiate_Property():
    with pytest.raises(TypeError):
        _ = Property()


def test_Property_abstract_methods():
    class FakeProperty(Property):
        def is_trivial(self):
            return super().is_trivial()

        def validate_counter_example(self, cex):
            return super().validate_counter_example(cex)

    prop = FakeProperty()

    with pytest.raises(NotImplementedError):
        prop.is_trivial()

    with pytest.raises(NotImplementedError):
        prop.validate_counter_example(None)


def test_instantiate_Reduction():
    with pytest.raises(TypeError):
        _ = Property()


def test_Reduction_abstract_methods():
    class FakeReduction(Reduction):
        def reduce_property(self, expression):
            return super().reduce_property(expression)

    reduction = FakeReduction()

    with pytest.raises(NotImplementedError):
        reduction.reduce_property(Constant(True))
