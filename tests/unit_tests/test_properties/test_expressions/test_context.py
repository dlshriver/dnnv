from dnnv.properties.expressions import Symbol
from dnnv.properties.expressions.context import *


def test_get_context():
    ctx = get_context()
    assert ctx is not None
    assert ctx is Context._current_context
    assert get_context() is ctx


def test_context_manager():
    ctx = get_context()
    assert ctx is not None
    assert get_context() is ctx

    with Context() as new_ctx:
        assert new_ctx is not ctx
        assert get_context() is new_ctx

        with new_ctx as same_ctx:
            assert same_ctx is new_ctx
            assert get_context() is same_ctx

        with ctx as old_ctx:
            assert old_ctx is not new_ctx
            assert old_ctx is ctx
            assert get_context() is old_ctx

    assert get_context() is ctx


def test_repr():
    ctx = get_context()
    assert repr(ctx) == "Context(id=0, prev_contexts=[])"

    with Context() as new_ctx:
        assert repr(new_ctx) == "Context(id=1, prev_contexts=[Context(id=0)])"

        with new_ctx as same_ctx:
            assert (
                repr(same_ctx)
                == "Context(id=1, prev_contexts=[Context(id=0), Context(id=1)])"
            )

        with ctx as old_ctx:
            assert repr(old_ctx) == "Context(id=0, prev_contexts=[Context(id=1)])"

    assert repr(ctx) == "Context(id=0, prev_contexts=[])"


def test_reset():
    ctx = get_context()
    a = Symbol("a")
    b = Symbol("b")
    assert Symbol in ctx._instance_cache
    assert len(ctx._instance_cache[Symbol]) == 2

    ctx.reset()
    assert len(ctx._instance_cache) == 0
