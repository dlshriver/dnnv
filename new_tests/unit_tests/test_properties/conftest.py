import pytest


@pytest.fixture(autouse=True)
def fresh_property_context():
    from dnnv.properties.context import Context

    Context._current_context = Context()
    yield
