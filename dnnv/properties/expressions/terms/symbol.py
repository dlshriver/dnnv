from __future__ import annotations

from typing import Optional, Union

from ...errors import NonConcreteExpressionError
from ..context import Context
from ..utils import empty_value
from .base import Term
from .constant import Constant


class Symbol(Term):
    def __init__(self, identifier: Union[Constant, str], ctx: Optional[Context] = None):
        super().__init__(ctx=ctx)
        if isinstance(identifier, Constant):
            self.identifier = identifier.value
        else:
            self.identifier = identifier
        self._value = empty_value

    def build_identifier(identifier, *args, **kwargs):
        if isinstance(identifier, Constant):
            identifier = identifier.value
        if not isinstance(identifier, str):
            raise TypeError("Argument 'identifier' for Symbol must be of type str")
        return identifier

    @property
    def is_concrete(self):
        return self._value is not empty_value

    @property
    def value(self):
        if not self.is_concrete:
            raise NonConcreteExpressionError(
                "Cannot get value of non-concrete expression."
            )
        return self._value

    def __hash__(self):
        return super().__hash__() * hash(self.identifier)

    def __repr__(self):
        return f"Symbol({self.identifier!r})"

    def __str__(self):
        return self.identifier


__all__ = ["Symbol"]
