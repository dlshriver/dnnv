from __future__ import annotations

from typing import Any, Hashable, Optional

from ..arithmetic import ArithmeticExpression
from ..call import CallableExpression
from ..context import Context, get_context
from ..logic import LogicalExpression


class _CachedType(type):
    def __call__(
        cls: _CachedType,
        *args: Any,
        ctx: Optional[Context] = None,
        **kwargs: Any,
    ) -> Any:
        ctx = ctx or get_context()
        if cls not in ctx._instance_cache:
            ctx._instance_cache[cls] = {}
        identifier = cls.build_identifier(*args, **kwargs)
        if identifier not in ctx._instance_cache[cls]:
            ctx._instance_cache[cls][identifier] = super().__call__(
                *args, ctx=ctx, **kwargs
            )
        return ctx._instance_cache[cls][identifier]

    def build_identifier(*args, **kwargs) -> Hashable:
        identifier = args + tuple(kv for kv in kwargs.items())
        return identifier


class Term(
    CallableExpression,
    ArithmeticExpression,
    LogicalExpression,
    metaclass=_CachedType,
):
    pass


__all__ = ["Term"]
