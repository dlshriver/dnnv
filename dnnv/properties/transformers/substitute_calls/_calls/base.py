from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Collection, Optional

from ....expressions import CallableExpression, Expression


class FunctionSubstitutor:
    __matches__: Collection[Callable]

    @classmethod
    def lookup(cls, f: Callable) -> Optional[FunctionSubstitutor]:
        for c in cls.__subclasses__():
            if f in c.__matches__:
                return c()
        return None

    @abstractmethod
    def __call__(
        self, f: CallableExpression, *args: Expression, **kwargs: Expression
    ) -> Expression:
        pass

    @staticmethod
    def substitute_Equal(
        a: Expression, b: Expression, form: Optional[str] = None
    ) -> Expression:
        return NotImplemented

    @staticmethod
    def substitute_NotEqual(
        a: Expression, b: Expression, form: Optional[str] = None
    ) -> Expression:
        return NotImplemented


__all__ = ["FunctionSubstitutor"]
