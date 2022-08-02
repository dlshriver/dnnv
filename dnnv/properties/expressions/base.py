from __future__ import annotations

import typing
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from ..errors import NonConcreteExpressionError
from .context import *
from .utils import empty_value

if typing.TYPE_CHECKING:  # pragma: no cover
    from .attribute import Attribute
    from .logic import (
        Equal,
        GreaterThan,
        GreaterThanOrEqual,
        LessThan,
        LessThanOrEqual,
        NotEqual,
    )
    from .subscript import Subscript
    from .terms import Constant, Network, Symbol

ExpressionType = TypeVar("ExpressionType", bound="Expression")


class Expression:
    def __new__(cls, *args, **kwargs):
        if cls is Expression:
            raise TypeError("Expression may not be instantiated")
        return object.__new__(cls)

    def __init__(self, ctx: Optional[Context] = None):
        self.ctx: Context = ctx or get_context()
        self._hash_cache_base: Optional[int] = None
        self.__FLAG_is_concrete = False
        self._iter_cache: Dict[
            Tuple[int, bool], Tuple[List[Expression], Iterator[Expression], bool]
        ] = {}

    def __bool__(self):
        if self.is_concrete:
            return bool(self.value)
        raise ValueError("The truth value of a non-concrete expression is ambiguous")

    def __hash__(self) -> int:
        if self._hash_cache_base is None:
            exprs_hash = 1
            for expr in self.iter(max_depth=1, include_self=False):
                exprs_hash *= hash(expr)
            self._hash_cache_base = hash(self.__class__) * exprs_hash
        return self._hash_cache_base

    def __getattr__(self, name) -> Union[Attribute, Constant]:
        from .attribute import Attribute
        from .terms import Constant

        if isinstance(name, str) and name.startswith("__"):
            # This case allows access to flags like `__FLAG_is_concrete`
            return super().__getattribute__(name)
        with self.ctx:
            if isinstance(name, Expression):
                return Attribute(self, name)
            return Attribute(self, Constant(name))

    def __getitem__(self, index) -> Union[Constant, Subscript]:
        from .slices import Slice
        from .subscript import Subscript
        from .terms import Constant

        with self.ctx:
            if not isinstance(index, Expression) and self.is_concrete:
                return Constant(self.value[index])
            if isinstance(index, slice):
                index = Slice(index.start, index.stop, index.step)
            if isinstance(index, Expression):
                if index.is_concrete and self.is_concrete:
                    return Constant(self.value[index.value])
                return Subscript(self, index)
            return Subscript(self, Constant(index))

    def __eq__(self, other) -> Equal:  # type: ignore
        from .logic import Equal

        with self.ctx:
            if isinstance(other, Expression):
                return Equal(self, other)
            from .terms import Constant

            return Equal(self, Constant(other))

    def __ge__(self, other) -> GreaterThanOrEqual:
        from .logic import GreaterThanOrEqual

        with self.ctx:
            if isinstance(other, Expression):
                return GreaterThanOrEqual(self, other)
            from .terms import Constant

            return GreaterThanOrEqual(self, Constant(other))

    def __gt__(self, other) -> GreaterThan:
        from .logic import GreaterThan

        with self.ctx:
            if isinstance(other, Expression):
                return GreaterThan(self, other)
            from .terms import Constant

            return GreaterThan(self, Constant(other))

    def __le__(self, other) -> LessThanOrEqual:
        from .logic import LessThanOrEqual

        with self.ctx:
            if isinstance(other, Expression):
                return LessThanOrEqual(self, other)
            from .terms import Constant

            return LessThanOrEqual(self, Constant(other))

    def __lt__(self, other) -> LessThan:
        from .logic import LessThan

        with self.ctx:
            if isinstance(other, Expression):
                return LessThan(self, other)
            from .terms import Constant

            return LessThan(self, Constant(other))

    def __ne__(self, other) -> NotEqual:  # type: ignore
        from .logic import NotEqual

        with self.ctx:
            if isinstance(other, Expression):
                return NotEqual(self, other)
            from .terms import Constant

            return NotEqual(self, Constant(other))

    def iter(self, max_depth=-1, include_self=True) -> Iterator[Expression]:
        if (max_depth, include_self) not in self._iter_cache:
            cached_seq: List[Expression] = []
            first_run = True
            self._iter_cache[max_depth, include_self] = (
                cached_seq,
                iter(()),
                first_run,
            )
            iterator = self.iter(max_depth=max_depth, include_self=include_self)
            self._iter_cache[max_depth, include_self] = (
                cached_seq,
                iterator,
                first_run,
            )
            for expr in iterator:
                cached_seq.append(expr)
                yield expr
            return
        elif not self._iter_cache[max_depth, include_self][2]:
            cached_seq = self._iter_cache[max_depth, include_self][0]
            yield from cached_seq
            for expr in self._iter_cache[max_depth, include_self][1]:
                cached_seq.append(expr)
                yield expr
            return
        cached_seq, iterator, _ = self._iter_cache[max_depth, include_self]
        self._iter_cache[max_depth, include_self] = (cached_seq, iterator, False)
        if include_self:
            yield self
        if max_depth == 0:
            return
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, Expression):
                yield from value.iter(max_depth=max_depth - 1)
            elif isinstance(value, (list, tuple, set)):
                for sub_value in value:
                    if isinstance(sub_value, Expression):
                        yield from sub_value.iter(max_depth=max_depth - 1)
            elif isinstance(value, dict):
                for key, sub_value in value.items():
                    if isinstance(key, Expression):
                        yield from key.iter(max_depth=max_depth - 1)
                    if isinstance(sub_value, Expression):
                        yield from sub_value.iter(max_depth=max_depth - 1)

    def canonical(self) -> Expression:
        from dnnv.properties.transformers import CanonicalTransformer

        with self.ctx:
            return CanonicalTransformer().visit(self)

    def concretize(
        self: ExpressionType, *args, **kwargs
    ) -> Union[ExpressionType, Symbol]:
        from .terms import Symbol

        nargs = len(args)
        if nargs > 0:
            if not isinstance(self, Symbol):
                raise ValueError(
                    f"Cannot concretize expression of type '{type(self).__name__}'"
                )
            if len(kwargs) > 0:
                raise ValueError(
                    "Cannot specify both keyword and positional arguments for method 'concretize'"
                )
            if nargs > 1:
                raise ValueError(
                    "Method 'concretize' expects at most 1 positional argument"
                )
            self._value = args[0]
            return self
        elif len(kwargs) == 0:
            raise ValueError("Method 'concretize' expects at least 1 argument")

        symbols = {
            child.identifier: child
            for child in self.iter()
            if isinstance(child, Symbol)
        }
        for name, value in kwargs.items():
            if name not in symbols:
                raise ValueError(f"Unknown identifier {name!r} for method 'concretize")
            symbols[name]._value = value
            symbols[name].__FLAG_is_concrete = True
        return self

    @property
    def is_concrete(self) -> bool:
        if self.__FLAG_is_concrete:
            return True
        try:
            _ = self.value
        except NonConcreteExpressionError:
            return False
        self.__FLAG_is_concrete = True
        return True

    def is_equivalent(self, other: Expression) -> bool:
        if self is other:
            return True
        try:
            return bool(np.all(self.value == other.value))
        except NonConcreteExpressionError:
            return False

    @property
    def networks(self) -> Set[Network]:
        from .terms import Network

        return set(expr for expr in self.iter() if isinstance(expr, Network))

    def propagate_constants(self) -> Expression:
        from dnnv.properties.transformers import PropagateConstants

        with self.ctx:
            return PropagateConstants().visit(self)

    @property
    def value(self) -> Any:
        raise NonConcreteExpressionError("Cannot get value of non-concrete expression")

    @property
    def variables(self) -> Set[Symbol]:
        from .terms import Symbol

        return set(
            expr
            for expr in self.iter()
            if isinstance(expr, Symbol) and not expr.is_concrete
        )


class AssociativeExpression(Expression):
    BASE_VALUE: Any
    DOMINANT_VALUES: List = []
    OPERATOR: Callable
    OPERATOR_SYMBOL: str

    def __init__(self, *expr: Expression, ctx: Optional[Context] = None):
        super().__init__(ctx=ctx)
        self.expressions: List[Expression] = []
        for expression in expr:
            if isinstance(expression, self.__class__):
                self.expressions.extend(expression.expressions)
            else:
                self.expressions.append(expression)
        self._expression_set = set(self.expressions)
        self._value = empty_value

    def is_equivalent(self, other):
        if super().is_equivalent(other):
            return True
        if type(self) != type(other) or len(self.expressions) != len(other.expressions):
            return False
        assert isinstance(other, AssociativeExpression)
        for e1, e2 in zip(self.expressions, other.expressions):
            if e1 not in other._expression_set or e2 not in self._expression_set:
                return False
        return True

    @property
    def value(self):
        if self._value is empty_value:
            if len(self.expressions) > 0:
                self._value = reduce(
                    self.OPERATOR, (expr.value for expr in self.expressions)
                )
            else:
                self._value = reduce(
                    self.OPERATOR,
                    (expr.value for expr in self.expressions),
                    self.BASE_VALUE,
                )
        return self._value

    def __repr__(self):
        result_str = f", ".join(
            repr(expr) for expr in sorted(self.expressions, key=repr)
        )
        return f"{type(self).__name__}({result_str})"

    def __str__(self):
        result_str = f" {self.OPERATOR_SYMBOL} ".join(
            str(expr) for expr in self.expressions
        )
        return f"({result_str})"

    def __iter__(self):
        for expr in self.expressions:
            yield expr


class BinaryExpression(Expression):
    def __init__(
        self,
        expr1: Expression,
        expr2: Expression,
        *,
        ctx: Optional[Context] = None,
    ):
        super().__init__(ctx=ctx)
        self.expr1 = expr1
        self.expr2 = expr2
        self._value = empty_value

    def is_equivalent(self, other):
        if super().is_equivalent(other):
            return True
        if (
            type(self) == type(other)
            and self.expr1.is_equivalent(other.expr1)
            and self.expr2.is_equivalent(other.expr2)
        ):
            return True
        return False

    @property
    def value(self):
        if self._value is empty_value:
            self._value = self.OPERATOR(self.expr1.value, self.expr2.value)
        return self._value

    def __repr__(self):
        return f"{type(self).__name__}({self.expr1!r}, {self.expr2!r})"

    def __str__(self):
        return f"({self.expr1} {self.OPERATOR_SYMBOL} {self.expr2})"


class TernaryExpression(Expression):
    def __init__(
        self,
        expr1: Expression,
        expr2: Expression,
        expr3: Expression,
        *,
        ctx: Optional[Context] = None,
    ):
        super().__init__(ctx=ctx)
        self.expr1 = expr1
        self.expr2 = expr2
        self.expr3 = expr3
        self._value = empty_value

    def is_equivalent(self, other):
        if super().is_equivalent(other):
            return True
        if (
            type(self) == type(other)
            and self.expr1.is_equivalent(other.expr1)
            and self.expr2.is_equivalent(other.expr2)
            and self.expr3.is_equivalent(other.expr3)
        ):
            return True
        return False

    @property
    def value(self):
        if self._value is empty_value:
            self._value = self.OPERATOR(
                self.expr1.value, self.expr2.value, self.expr3.value
            )
        return self._value

    def __repr__(self):
        return f"{type(self).__name__}({self.expr1!r}, {self.expr2!r}, {self.expr3!r})"

    def __str__(self):
        return f"{type(self).__name__}({self.expr1}, {self.expr2}, {self.expr3})"


class UnaryExpression(Expression):
    def __init__(self, expr: Expression, *, ctx: Optional[Context] = None):
        super().__init__(ctx=ctx)
        self.expr = expr
        self._value = empty_value

    def is_equivalent(self, other):
        if super().is_equivalent(other):
            return True
        if type(self) == type(other) and self.expr.is_equivalent(other.expr):
            return True
        return False

    @property
    def value(self):
        if self._value is empty_value:
            self._value = self.OPERATOR(self.expr.value)
        return self._value

    def __repr__(self):
        return f"{type(self).__name__}({self.expr!r})"

    def __str__(self):
        return f"{self.OPERATOR_SYMBOL}{self.expr}"


__all__ = [
    "AssociativeExpression",
    "BinaryExpression",
    "Expression",
    "TernaryExpression",
    "UnaryExpression",
]
