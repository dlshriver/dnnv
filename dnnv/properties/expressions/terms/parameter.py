from __future__ import annotations

from typing import Callable, Optional, TypeVar, Union

from ..base import Expression
from ..context import Context
from .constant import Constant
from .symbol import Symbol
from .utils import get_callable_name

T = TypeVar("T")


class Parameter(Symbol):
    def __init__(
        self,
        identifier: Union[Constant, str],
        type: Union[Constant, Callable[[str], T]],
        default: Optional[Union[Expression, T]] = None,
        ctx: Optional[Context] = None,
    ):
        super().__init__(identifier, ctx=ctx)
        if isinstance(identifier, Constant):
            self.name = identifier.value
        else:
            self.name = identifier
        if isinstance(type, Constant):
            self.type: Callable[[str], T] = type.value
        else:
            self.type = type
        if default is None:
            self.default = default
        elif isinstance(default, Expression):
            self.default = default
        else:
            self.default = Constant(default)

    def __hash__(self):
        return super().__hash__() * hash(self.identifier) * hash(self.type)

    def __repr__(self):
        if self.is_concrete:
            return f"Parameter({self.name!r}, value={repr(self.value)})"
        return f"Parameter({self.name!r}, type={get_callable_name(self.type)}, default={self.default!r})"

    def __str__(self):
        if self.is_concrete:
            return repr(self.value)
        return f"Parameter({self.name!r}, type={get_callable_name(self.type)}, default={self.default!r})"


__all__ = ["Parameter"]
