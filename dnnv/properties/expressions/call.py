from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from .arithmetic import ArithmeticExpression
from .base import Expression
from .context import Context
from .logic import LogicalExpression
from .utils import empty_value


class CallableExpression(Expression):
    def __call__(self, *args: Expression, **kwargs: Expression) -> Call:
        return Call(self, args, kwargs, ctx=self.ctx)


class Call(CallableExpression, ArithmeticExpression, LogicalExpression):
    def __init__(
        self,
        function: CallableExpression,
        args: Tuple[Expression, ...],
        kwargs: Mapping[str, Expression],
        *,
        ctx: Optional[Context] = None,
    ):
        super().__init__(ctx=ctx)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._value: Any = empty_value
        self._hash_cache_call = None

    @property
    def value(self):
        if self._value is not empty_value:
            return self._value
        args = tuple(arg.value for arg in self.args)
        kwargs = {name: value.value for name, value in self.kwargs.items()}
        self._value = self.function.value(*args, **kwargs)
        return self._value

    def is_equivalent(self, other):
        if super().is_equivalent(other):
            return True
        elif (
            isinstance(other, Call)
            and self.function.is_equivalent(other.function)
            and all(
                arg1.is_equivalent(arg2) for arg1, arg2 in zip(self.args, other.args)
            )
            and all(
                self.kwargs[k].is_equivalent(other.kwargs[k])
                if k in self.kwargs and k in other.kwargs
                else False
                for k in set(self.kwargs.keys()).union(other.kwargs.keys())
            )
        ):
            return True
        return False

    def __hash__(self):
        if self._hash_cache_call is None:
            args_hash = 1
            for arg in self.args:
                args_hash *= hash(arg)
            for key, arg in self.kwargs.items():
                args_hash *= hash(key) * hash(arg)
            self._hash_cache_call = super().__hash__() * hash(self.function) * args_hash
        return self._hash_cache_call

    def __repr__(self):
        function_name = repr(self.function)
        args_str = ", ".join(repr(arg) for arg in self.args)
        kwargs_str = ", ".join(
            f"{name}={value!r}" for name, value in self.kwargs.items()
        )
        if args_str and kwargs_str:
            return f"{function_name}({args_str}, {kwargs_str})"
        elif args_str:
            return f"{function_name}({args_str})"
        elif kwargs_str:
            return f"{function_name}({kwargs_str})"
        return f"{function_name}()"

    def __str__(self):
        function_name = str(self.function)
        args_str = ", ".join(str(arg) for arg in self.args)
        kwargs_str = ", ".join(f"{name}={value}" for name, value in self.kwargs.items())
        if args_str and kwargs_str:
            return f"{function_name}({args_str}, {kwargs_str})"
        elif args_str:
            return f"{function_name}({args_str})"
        elif kwargs_str:
            return f"{function_name}({kwargs_str})"
        return f"{function_name}()"


__all__ = ["Call", "CallableExpression"]
