import inspect
import numpy as np
import types

from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from .context import Context, get_context

T = TypeVar("T")


class Expression:
    def __new__(cls, *args, **kwargs):
        if cls is Expression:
            raise TypeError("Expression may not be instantiated")
        return object.__new__(cls)

    def __init__(self, ctx: Optional[Context] = None):
        self.ctx = ctx or get_context()

    def concretize(self, **kwargs) -> "Expression":
        symbols = {s.identifier: s for s in find_symbols(self)}
        for name, value in kwargs.items():
            if name not in symbols:
                raise ValueError(f"Unknown identifier: {name!r}")
            symbols[name].concretize(value)
        return self

    def canonical(self) -> "Expression":
        from .transformers import Canonical

        with self.ctx:
            return Canonical().visit(self)

    def propagate_constants(self) -> "Expression":
        from .transformers import PropagateConstants

        with self.ctx:
            return PropagateConstants().visit(self)

    def to_cnf(self) -> "Expression":
        from .transformers import ToCNF

        with self.ctx:
            return ToCNF().visit(self)

    @property
    def is_concrete(self) -> bool:
        return len(self.variables) == 0

    @property
    def value(self):
        if not self.is_concrete:
            raise ValueError("Cannot get value of non-concrete expression.")
        else:
            value = self.propagate_constants()
            if isinstance(value, Constant):
                return value.value
            raise NotImplementedError(
                f"Method 'value' is not implemented for type '{type(self).__name__}'"
            )

    @property
    def networks(self) -> List["Network"]:
        return list(find_symbols(self, Network))

    @property
    def parameters(self) -> List["Parameter"]:
        return list(find_symbols(self, Parameter))

    @property
    def variables(self) -> List["Symbol"]:
        return [s for s in find_symbols(self) if not s.is_concrete]

    def __bool__(self):
        if self.is_concrete:
            return bool(self.value)
        return True

    def __hash__(self):
        return hash(self.__class__) * hash(repr(self))

    def __getattr__(self, name) -> Union["Attribute", "Constant"]:
        if isinstance(name, str) and name.startswith("__"):
            return super().__getattribute__(name)
        if isinstance(name, str) and self.is_concrete:
            return Constant(getattr(self.value, name))
        elif isinstance(name, Constant) and self.is_concrete:
            return Constant(getattr(self.value, name.value))
        elif isinstance(name, Expression):
            return Attribute(self, name)
        return Attribute(self, Constant(name))

    def __getitem__(self, index) -> Union["Constant", "Subscript"]:
        if not isinstance(index, Expression) and self.is_concrete:
            return Constant(self.value[index])
        if isinstance(index, slice):
            index = Slice(index.start, index.stop, index.step)
        if isinstance(index, Expression):
            if index.is_concrete and self.is_concrete:
                return Constant(self.value[index.value])
            return Subscript(self, index)
        return Subscript(self, Constant(index))

    def __add__(self, other) -> "Add":
        if isinstance(other, Expression):
            return Add(self, other)
        return Add(self, Constant(other))

    def __radd__(self, other) -> "Add":
        if isinstance(other, Expression):
            return Add(other, self)
        return Add(Constant(other), self)

    def __sub__(self, other) -> "Subtract":
        if isinstance(other, Expression):
            return Subtract(self, other)
        return Subtract(self, Constant(other))

    def __rsub__(self, other) -> "Subtract":
        if isinstance(other, Expression):
            return Subtract(other, self)
        return Subtract(Constant(other), self)

    def __mul__(self, other) -> "Multiply":
        if isinstance(other, Expression):
            return Multiply(self, other)
        return Multiply(self, Constant(other))

    def __rmul__(self, other) -> "Multiply":
        if isinstance(other, Expression):
            return Multiply(other, self)
        return Multiply(Constant(other), self)

    def __truediv__(self, other) -> "Divide":
        if isinstance(other, Expression):
            return Divide(self, other)
        return Divide(self, Constant(other))

    def __rtruediv__(self, other) -> "Divide":
        if isinstance(other, Expression):
            return Divide(other, self)
        return Divide(Constant(other), self)

    def __neg__(self) -> "Negation":
        return Negation(self)

    def __eq__(self, other) -> "Equal":
        if isinstance(other, Expression):
            return Equal(self, other)
        return Equal(self, Constant(other))

    def __ne__(self, other) -> "NotEqual":
        if isinstance(other, Expression):
            return NotEqual(self, other)
        return NotEqual(self, Constant(other))

    def __ge__(self, other) -> "GreaterThanOrEqual":
        if isinstance(other, Expression):
            return GreaterThanOrEqual(self, other)
        return GreaterThanOrEqual(self, Constant(other))

    def __gt__(self, other) -> "GreaterThan":
        if isinstance(other, Expression):
            return GreaterThan(self, other)
        return GreaterThan(self, Constant(other))

    def __le__(self, other) -> "LessThanOrEqual":
        if isinstance(other, Expression):
            return LessThanOrEqual(self, other)
        return LessThanOrEqual(self, Constant(other))

    def __lt__(self, other) -> "LessThan":
        if isinstance(other, Expression):
            return LessThan(self, other)
        return LessThan(self, Constant(other))

    def __and__(self, other) -> "And":
        return And(self, other)

    def __rand__(self, other) -> "And":
        return And(other, self)

    def __or__(self, other) -> "Or":
        return Or(self, other)

    def __ror__(self, other) -> "Or":
        return Or(other, self)

    def __invert__(self) -> "Not":
        return Not(self)

    def __call__(
        self, *args: "Expression", **kwargs: "Expression"
    ) -> Union["Constant", "FunctionCall"]:
        if (
            self.is_concrete
            and all(not isinstance(arg, Expression) or arg.is_concrete for arg in args)
            and all(
                not isinstance(v, Expression) or v.is_concrete for v in kwargs.values()
            )
        ):
            args_values = [
                arg.value if isinstance(arg, Expression) else arg for arg in args
            ]
            kwargs_values = {
                k: v.value if isinstance(v, Expression) else v
                for k, v in kwargs.items()
            }
            return Constant(self.value(*args_values, **kwargs_values))
        return FunctionCall(self, args, kwargs)


def hidesignature(func):
    # hides the function signature from mypy
    # used in CachedExpression to hide signatures of:
    #   - initialize
    #   - build_identifier
    # which have more specific signatures in subclasses
    # TODO : can we find a better way to do this?
    return func


class CachedExpression(Expression):
    __initialized = False

    def __new__(cls, *args, ctx: Optional[Context] = None, **kwargs):
        inspect.getcallargs(cls.initialize, None, *args, *kwargs)
        ctx = ctx or get_context()
        if cls not in ctx._instance_cache:
            ctx._instance_cache[cls] = {}
        identifier = cls.build_identifier(*args, **kwargs)
        if identifier not in ctx._instance_cache[cls]:
            ctx._instance_cache[cls][identifier] = super().__new__(cls)
            ctx._instance_cache[cls][identifier].__initialized = False
            ctx._instance_cache[cls][identifier].__newargs = (args, ctx, kwargs)
        return ctx._instance_cache[cls][identifier]

    def __init__(self, *args, ctx: Optional[Context] = None, **kwargs):
        super().__init__(ctx=ctx)
        if self.__initialized:
            return
        self.initialize(*args, **kwargs)
        self.__initialized = True

    def __getnewargs_ex__(self):
        args, ctx, kwargs = self.__newargs
        kwargs["ctx"] = ctx
        return args, kwargs

    @hidesignature
    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError()

    @hidesignature
    @classmethod
    @abstractmethod
    def build_identifier(cls, *args, **kwargs) -> str:
        raise NotImplementedError()


class Constant(CachedExpression):
    def initialize(self, value: Any):
        self._value = value
        self._id = len(self.ctx._instance_cache[Constant])

    @classmethod
    def build_identifier(cls, value: Any):
        if isinstance(value, Constant):
            value = value._value
        value_type = type(value)
        try:
            value_hash = hash(value)
            value_identifier = value
        except:
            value_identifier = id(value)
        return value_type, value_identifier

    @property
    def is_concrete(self):
        return True

    @property
    def value(self):
        if isinstance(self._value, list):
            return [
                v.value if isinstance(v, Expression) and v.is_concrete else v
                for v in self._value
            ]
        elif isinstance(self._value, set):
            return set(
                v.value if isinstance(v, Expression) and v.is_concrete else v
                for v in self._value
            )
        elif isinstance(self._value, tuple):
            return tuple(
                v.value if isinstance(v, Expression) and v.is_concrete else v
                for v in self._value
            )
        return self._value

    def __bool__(self):
        return bool(self.value)

    def __repr__(self):
        value = self.value
        if isinstance(value, (str, int, float, tuple, list, set, dict)):
            return repr(value)
        elif isinstance(value, slice):
            start = value.start if value.start is not None else ""
            stop = value.stop if value.stop is not None else ""
            if value.step is not None:
                return f"{start}:{stop}:{value.step}"
            return f"{start}:{stop}"
        return f"{type(value)}(id={hex(self._id)})"

    def __str__(self):
        value = self.value
        if isinstance(value, np.ndarray):
            return "".join(
                np.array2string(
                    value, max_line_width=np.inf, precision=3, threshold=5, edgeitems=2
                ).split("\n")
            ).replace("  ", " ")
        elif isinstance(value, slice):
            start = value.start if value.start is not None else ""
            stop = value.stop if value.stop is not None else ""
            if value.step is not None:
                return f"{start}:{stop}:{value.step}"
            return f"{start}:{stop}"
        return str(value)


class Symbol(CachedExpression):
    def initialize(self, identifier: Union[Constant, str]):
        self.identifier = str(identifier)
        self._value = None

    @classmethod
    def build_identifier(cls, identifier: Union[Constant, str]):
        if isinstance(identifier, Constant):
            identifier = identifier.value
        if not isinstance(identifier, str):
            raise TypeError(
                f"Argument identifier for {cls.__name__} must be of type str"
            )
        return identifier

    @property
    def value(self):
        if not self.is_concrete:
            raise ValueError("Cannot get value of non-concrete symbol.")
        return self._value

    @property
    def is_concrete(self):
        return self._value is not None

    def concretize(self, value):
        self._value = value

    def __bool__(self):
        if self.is_concrete:
            return bool(self.value)
        return True

    def __repr__(self):
        return f"Symbol({self.identifier!r})"

    def __str__(self):
        return self.identifier


def _get_function_name(function: Callable) -> str:
    if isinstance(function, types.LambdaType) and (
        function.__name__ == "<lambda>" or function.__qualname__ == "<lambda>"
    ):
        return f"{function.__module__}.{function.__name__}"
    elif isinstance(function, types.BuiltinFunctionType):
        return f"{function.__name__}"
    elif isinstance(function, types.FunctionType):
        return f"{function.__module__}.{function.__name__}"
    elif isinstance(function, types.MethodType):
        if function.__self__.__module__ == "__main__":
            return f"{function.__self__.__class__.__name__}.{function.__name__}"
        return f"{function.__self__.__module__}.{function.__self__.__class__.__name__}.{function.__name__}"
    elif isinstance(function, np.ufunc):
        return f"numpy.{function.__name__}"
    elif isinstance(function, type) and callable(function):
        return f"{function.__module__}.{function.__name__}"
    else:
        raise ValueError(f"Unsupported function type: {type(function)}")


def _symbol_from_callable(symbol: Callable):
    if isinstance(symbol, Expression):
        return symbol
    name = _get_function_name(symbol)
    new_symbol = Symbol(name)
    new_symbol.concretize(symbol)
    return new_symbol


def find_symbols(phi: Expression, symbol_class: Type[Symbol] = Symbol):
    symbols = set()
    for value in phi.__dict__.values():
        if isinstance(value, symbol_class):
            symbols.add(value)
        if isinstance(value, Expression):
            symbols = symbols.union(find_symbols(value, symbol_class))
        elif isinstance(value, (list, tuple, set)):
            for sub_value in value:
                if isinstance(sub_value, symbol_class):
                    symbols.add(sub_value)
                if isinstance(sub_value, Expression):
                    symbols = symbols.union(find_symbols(sub_value, symbol_class))
        elif isinstance(value, dict):
            for key, sub_value in value.items():
                if isinstance(key, symbol_class):
                    symbols.add(key)
                if isinstance(key, Expression):
                    symbols = symbols.union(find_symbols(key, symbol_class))
                if isinstance(sub_value, symbol_class):
                    symbols.add(sub_value)
                if isinstance(sub_value, Expression):
                    symbols = symbols.union(find_symbols(sub_value, symbol_class))
    return symbols


class Parameter(Symbol):
    def initialize(
        self,
        identifier: Union[Constant, str],
        type: Type[T],
        default: Optional[Union[Constant, T]] = None,
    ):
        super().initialize(identifier)
        self.name = str(identifier)
        self.type = type
        if isinstance(default, Constant):
            self.default = default.value
        else:
            self.default = default

    @classmethod
    def build_identifier(
        cls,
        identifier: Union[Constant, str],
        type: Type[T],
        default: Optional[Union[Constant, T]] = None,
    ):
        if isinstance(identifier, Constant):
            identifier = identifier.value
        if not isinstance(identifier, str):
            raise TypeError(
                f"Argument identifier for {cls.__name__} must be of type str"
            )
        return identifier

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_concrete:
            return repr(self.value)
        return f"Parameter({self.name!r}, type={self.type.__name__}, default={self.default!r})"


class Network(Symbol):
    def __new__(cls, identifier: Union[Constant, str] = "N", *args, **kwargs):
        return super().__new__(cls, identifier, *args, **kwargs)

    def __init__(self, identifier: Union[Constant, str] = "N", *args, **kwargs):
        return super().__init__(identifier, *args, **kwargs)

    def __getitem__(self, item):
        if self.is_concrete and isinstance(item, Constant):
            new_network = Network(f"{self}[{item!r}]")
            new_network.concretize(self.value[item.value])
            return new_network
        elif self.is_concrete and isinstance(item, int):
            new_network = Network(f"{self}[{item}]")
            new_network.concretize(self.value[item])
            return new_network
        elif self.is_concrete and isinstance(item, slice):
            start = item.start or ""
            stop = item.stop or ""
            s = f"{start}:{stop}"
            if item.step is not None:
                s = f"{s}:{item.step}"
            new_network = Network(f"{self}[{s}]")
            new_network.concretize(self.value[item])
            return new_network
        return super().__getitem__(item)

    def __repr__(self):
        return f"Network({self.identifier!r})"

    def compose(self, other: "Network"):
        if not isinstance(other, Network):
            raise ValueError("Networks can only be composed with other networks.")
        if self.is_concrete and other.is_concrete:
            new_network = Network(f"{self}○{other}")
            op_graph = self.value.compose(other.value)
            new_network.concretize(op_graph)
            return new_network
        raise ValueError("Network is not concrete.")


class Image(Expression):
    def __init__(self, path: Union[Expression, str], *, ctx: Optional[Context] = None):
        super().__init__(ctx=ctx)
        self.path = path

    @classmethod
    def load(cls, path: Expression):
        if not isinstance(path, Constant):
            return Image(path)
        # TODO : handle other image formats
        img = np.load(path.value)[None, :].astype(np.float32)
        return Constant(img)

    @property
    def value(self):
        if self.is_concrete:
            return self.load(self.path).value
        return super().value

    def __repr__(self):
        return f"Image({self.path!r})"

    def __str__(self):
        return f"Image({self.path})"


class FunctionCall(Expression):
    def __init__(
        self,
        function: Expression,
        args: Tuple[Expression, ...],
        kwargs: Dict[str, Expression],
        *,
        ctx: Optional[Context] = None,
    ):
        super().__init__(ctx=ctx)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._value = None

    @property
    def value(self):
        if self.is_concrete:
            if getattr(self.function.value, "__func__", None) == Network.compose:
                self._value = self.function.value(*self.args, **self.kwargs)
            elif self._value is None:
                args = tuple(arg.value for arg in self.args)
                kwargs = {name: value.value for name, value in self.kwargs.items()}
                self._value = self.function.value(*args, **kwargs)
            return self._value
        return super().value

    def __repr__(self):
        if not isinstance(self.function, Constant):
            function_name = repr(self.function)
        else:
            function_name = _get_function_name(self.function.value)
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
        if not isinstance(self.function, Constant):
            function_name = str(self.function)
        else:
            function_name = _get_function_name(self.function.value)
        args_str = ", ".join(str(arg) for arg in self.args)
        kwargs_str = ", ".join(f"{name}={value}" for name, value in self.kwargs.items())
        if args_str and kwargs_str:
            return f"{function_name}({args_str}, {kwargs_str})"
        elif args_str:
            return f"{function_name}({args_str})"
        elif kwargs_str:
            return f"{function_name}({kwargs_str})"
        return f"{function_name}()"


class AssociativeExpression(Expression):
    def __init__(self, *expr: Expression, ctx: Optional[Context] = None):
        super().__init__(ctx=ctx)
        self.expressions = []  # type: List[Expression]
        for expression in expr:
            if isinstance(expression, self.__class__):
                self.expressions.extend(expression.expressions)
            else:
                self.expressions.append(expression)

    def __repr__(self):
        result_str = f", ".join(
            repr(expr) for expr in sorted(self.expressions, key=repr)
        )
        return f"{type(self).__name__}({result_str})"

    def __str__(self):
        result_str = f" {self.OPERATOR} ".join(str(expr) for expr in self.expressions)
        return f"({result_str})"

    def __iter__(self):
        for expr in self.expressions:
            yield expr


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

    def __repr__(self):
        return f"{type(self).__name__}({self.expr1!r}, {self.expr2!r}, {self.expr3!r})"

    def __str__(self):
        return f"{type(self).__name__}({self.expr1}, {self.expr2}, {self.expr3})"


class BinaryExpression(Expression):
    def __init__(
        self, expr1: Expression, expr2: Expression, *, ctx: Optional[Context] = None
    ):
        super().__init__(ctx=ctx)
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return f"{type(self).__name__}({self.expr1!r}, {self.expr2!r})"

    def __str__(self):
        return f"({self.expr1} {self.OPERATOR} {self.expr2})"


class UnaryExpression(Expression):
    def __init__(self, expr: Expression, *, ctx: Optional[Context] = None):
        super().__init__(ctx=ctx)
        self.expr = expr

    def __repr__(self):
        return f"{type(self).__name__}({self.expr!r})"

    def __str__(self):
        return f"{self.OPERATOR}{self.expr}"


class Attribute(BinaryExpression):
    @property
    def expr(self):
        return self.expr1

    @property
    def name(self):
        return self.expr2

    @property
    def value(self):
        if self.is_concrete:
            return getattr(self.expr.value, self.index.value)
        return super().value

    def __repr__(self):
        return f"{self.expr!r}.{self.name!r}"

    def __str__(self):
        return f"{self.expr}.{self.name}"


class Slice(TernaryExpression):
    def __init__(
        self, start: Any, stop: Any, step: Any, *, ctx: Optional[Context] = None
    ):
        start = start if isinstance(start, Expression) else Constant(start)
        stop = stop if isinstance(stop, Expression) else Constant(stop)
        step = step if isinstance(step, Expression) else Constant(step)
        super().__init__(start, stop, step, ctx=ctx)

    @property
    def start(self):
        return self.expr1

    @property
    def stop(self):
        return self.expr2

    @property
    def step(self):
        return self.expr3

    @property
    def value(self):
        if self.is_concrete:
            start = self.start.value
            stop = self.stop.value
            step = self.step.value
            return slice(start, stop, step)
        return super().value

    def __repr__(self):
        start = "" if self.start.value is None else self.start
        stop = "" if self.stop.value is None else self.stop
        step = None if self.step.value is None else self.step
        if step is None:
            return f"{start!r}:{stop!r}"
        return f"{start!r}:{stop!r}:{step!r}"

    def __str__(self):
        start = "" if self.start.value is None else self.start
        stop = "" if self.stop.value is None else self.stop
        step = None if self.step.value is None else self.step
        if step is None:
            return f"{start}:{stop}"
        return f"{start}:{stop}:{step}"


class Subscript(BinaryExpression):
    @property
    def expr(self):
        return self.expr1

    @property
    def index(self):
        return self.expr2

    @property
    def value(self):
        if self.is_concrete:
            return self.expr.value[self.index.value]
        return super().value

    def __repr__(self):
        return f"{self.expr!r}[{self.index!r}]"

    def __str__(self):
        return f"{self.expr}[{self.index}]"


class IfThenElse(TernaryExpression):
    @property
    def condition(self):
        return self.expr1

    @property
    def t_expr(self):
        return self.expr2

    @property
    def f_expr(self):
        return self.expr3

    @property
    def value(self):
        if self.is_concrete:
            if self.condition.value:
                return self.t_expr.value
            else:
                return self.f_expr.value
        return super().value


class ArithmeticExpression(Expression):
    def __call__(self, *args: "Expression", **kwargs: "Expression"):
        raise ValueError("Arithmetic expressions are not callable.")


class Negation(UnaryExpression, ArithmeticExpression):
    OPERATOR = "-"

    def __neg__(self):
        return self.expr


class Add(AssociativeExpression, ArithmeticExpression):
    OPERATOR = "+"


class Subtract(BinaryExpression, ArithmeticExpression):
    OPERATOR = "-"


class Multiply(AssociativeExpression, ArithmeticExpression):
    OPERATOR = "*"


class Divide(BinaryExpression, ArithmeticExpression):
    OPERATOR = "/"


class LogicalExpression(Expression):
    def __invert__(self):
        return Not(self)

    def __call__(self, *args: "Expression", **kwargs: "Expression"):
        raise ValueError("Logical expressions are not callable.")


class Not(UnaryExpression, LogicalExpression):
    OPERATOR = "~"

    def __invert__(self):
        return self.expr


class Equal(BinaryExpression, LogicalExpression):
    OPERATOR = "=="

    def __invert__(self):
        return NotEqual(self.expr1, self.expr2)

    def __bool__(self):
        return repr(self.expr1) == repr(self.expr2)


class NotEqual(BinaryExpression, LogicalExpression):
    OPERATOR = "!="

    def __invert__(self):
        return Equal(self.expr1, self.expr2)

    def __bool__(self):
        return repr(self.expr1) != repr(self.expr2)


class LessThan(BinaryExpression, LogicalExpression):
    OPERATOR = "<"

    def __invert__(self):
        return GreaterThanOrEqual(self.expr1, self.expr2)


class LessThanOrEqual(BinaryExpression, LogicalExpression):
    OPERATOR = "<="

    def __invert__(self):
        return GreaterThan(self.expr1, self.expr2)


class GreaterThan(BinaryExpression, LogicalExpression):
    OPERATOR = ">"

    def __invert__(self):
        return LessThanOrEqual(self.expr1, self.expr2)


class GreaterThanOrEqual(BinaryExpression, LogicalExpression):
    OPERATOR = ">="

    def __invert__(self):
        return LessThan(self.expr1, self.expr2)


class Or(AssociativeExpression, LogicalExpression):
    OPERATOR = "|"

    def __invert__(self):
        return And(*[~expr for expr in self.expressions])


class And(AssociativeExpression, LogicalExpression):
    OPERATOR = "&"

    def __invert__(self):
        return Or(*[~expr for expr in self.expressions])


class Implies(BinaryExpression, LogicalExpression):
    OPERATOR = "==>"

    def __invert__(self):
        return And(self.expr1, ~self.expr2)


class Quantifier(LogicalExpression):
    def __init__(
        self,
        variable: Symbol,
        formula: Union[Expression, Callable[[Symbol], Expression]],
        *,
        ctx: Optional[Context] = None,
    ):
        super().__init__(ctx=ctx)
        self.variable = variable
        if not isinstance(formula, Expression):
            formula = formula(variable)
        self.expression = formula

    def __repr__(self):
        return f"{type(self).__name__}({self.variable!r}, {self.expression!r})"

    def __str__(self):
        return f"{type(self).__name__}({self.variable}, {self.expression})"


class Forall(Quantifier):
    def __invert__(self):
        return Exists(self.variable, lambda _: ~self.expression)


class Exists(Quantifier):
    def __invert__(self):
        return Forall(self.variable, lambda _: ~self.expression)


argmax = np.argmax
argmin = np.argmin


def __argcmp_helper(cmp_fn, A, Ai, i=0) -> Union[Constant, IfThenElse]:
    if i == len(Ai) - 1:
        return Constant(i)
    cond = And(*[cmp_fn(A[Ai[i]], A[Ai[j]]) for j in range(len(Ai)) if j != i])
    return IfThenElse(cond, Constant(i), __argcmp_helper(cmp_fn, A, Ai, i + 1))


def __argcmp(
    cmp_fn: Callable[[Expression, Expression], Expression], A: Expression
) -> Union[Constant, IfThenElse]:
    if not isinstance(A, Expression):
        return Constant(np.argmax(A))
    elif A.is_concrete:
        return Constant(np.argmax(A.value))
    elif isinstance(A, FunctionCall) and isinstance(A.function, Network):
        if not A.function.is_concrete:
            raise RuntimeError(
                "argcmp can not be applied to outputs of non-concrete networks"
            )
        output_shape = A.function.value.output_shape[0]
        Ai = list(np.ndindex(output_shape))
        argcmp = __argcmp_helper(cmp_fn, A, Ai)
        return argcmp
    else:
        raise RuntimeError(f"Unsupported type for argcmp input: {type(A)}")


def __argcmp_eq(
    cmp_fn: Callable[[Expression, Expression], Expression],
    F: FunctionCall,
    E: Expression,
) -> Union[And, Or, Constant]:
    if len(F.args) > 1 or len(F.kwargs) != 0:
        raise RuntimeError("Too many arguments for argcmp")
    A = F.args[0]
    if A.is_concrete and E.is_concrete:
        return Constant(np.argmax(A.value) == E.value)
    elif isinstance(A, FunctionCall) and isinstance(A.function, Network):
        if not A.function.is_concrete:
            raise RuntimeError(
                "argcmp can not be applied to outputs of non-concrete networks"
            )
        output_shape = A.function.value.output_shape[0]
        Ai = list(np.ndindex(output_shape))
        if E.is_concrete:
            c = E.value
            if c > len(Ai):
                return Constant(False)
            return And(*[cmp_fn(A[Ai[c]], A[Ai[i]]) for i in range(len(Ai)) if i != c])
        return And(*[Implies(F == c, E == c) for c in range(len(Ai))])
    return NotImplemented


def __argcmp_neq(
    cmp_fn: Callable[[Expression, Expression], Expression],
    F: FunctionCall,
    E: Expression,
) -> Union[And, Or, Constant]:
    if len(F.args) > 1 or len(F.kwargs) != 0:
        raise RuntimeError("Too many arguments for argcmp")
    A = F.args[0]
    if A.is_concrete and E.is_concrete:
        return Constant(np.argmax(A.value) != E.value)
    elif isinstance(A, FunctionCall) and isinstance(A.function, Network):
        if not A.function.is_concrete:
            raise RuntimeError(
                "argcmp can not be applied to outputs of non-concrete networks"
            )
        output_shape = A.function.value.output_shape[0]
        Ai = list(np.ndindex(output_shape))
        if E.is_concrete:
            c = E.value
            return Or(*[~cmp_fn(A[Ai[c]], A[Ai[i]]) for i in range(len(Ai)) if i != c])
        return And(*[Or(F != c, E != c) for c in range(len(Ai))])
    return NotImplemented


def __abs(x) -> Union[Constant, IfThenElse]:
    if not isinstance(x, Expression):
        return Constant(abs(x))
    elif x.is_concrete:
        return Constant(abs(x.value))
    return IfThenElse(x >= 0, x, -x)


_BUILTIN_FUNCTION_TRANSFORMS = {
    (np.argmax, Equal): partial(__argcmp_eq, lambda a, b: a >= b),
    (np.argmin, Equal): partial(__argcmp_eq, lambda a, b: a <= b),
    (np.argmax, NotEqual): partial(__argcmp_neq, lambda a, b: a >= b),
    (np.argmin, NotEqual): partial(__argcmp_neq, lambda a, b: a <= b),
}  # type: Dict[Tuple[Callable, Type[Expression]], Callable[[FunctionCall, Expression], Expression]]

_BUILTIN_FUNCTIONS = {
    abs: __abs,
    np.abs: __abs,
    np.argmax: partial(__argcmp, lambda a, b: a >= b),
    np.argmin: partial(__argcmp, lambda a, b: a <= b),
}  # type: Dict[Callable, Callable[..., Expression]]

# TODO : organize this list better
__all__ = [
    "Expression",
    "Symbol",
    "Network",
    "Parameter",
    "Constant",
    "Image",
    "FunctionCall",
    "Attribute",
    "Subscript",
    "Slice",
    "IfThenElse",
    "Negation",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Not",
    "Equal",
    "NotEqual",
    "LessThan",
    "LessThanOrEqual",
    "GreaterThan",
    "GreaterThanOrEqual",
    "Or",
    "And",
    "Implies",
    "Quantifier",
    "Forall",
    "Exists",
    "AssociativeExpression",
    "TernaryExpression",
    "BinaryExpression",
    "UnaryExpression",
]
