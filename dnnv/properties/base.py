import numpy as np
import types

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from dnnv import logging


class Expression:
    def __new__(cls, *args, **kwargs):
        if cls is Expression:
            raise TypeError("Expression may not be instantiated")
        return object.__new__(cls)

    def concretize(self, **kwargs) -> "Expression":
        symbols = {s.identifier: s for s in find_symbols(self)}
        for name, value in kwargs.items():
            if name not in symbols:
                raise ValueError(f"Unknown identifier: {name!r}")
            symbols[name].concretize(value)
        return self

    def canonical(self) -> "Expression":
        from dnnv.properties.transformers import Canonical

        return Canonical().visit(self)

    def propagate_constants(self) -> "Expression":
        from dnnv.properties.transformers import PropagateConstants

        return PropagateConstants().visit(self)

    def to_cnf(self) -> "Expression":
        from dnnv.properties.transformers import ToCNF

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
            args = [arg.value if isinstance(arg, Expression) else arg for arg in args]
            kwargs = {
                k: v.value if isinstance(v, Expression) else v
                for k, v in kwargs.items()
            }
            return Constant(self.value(*args, **kwargs))
        return FunctionCall(self, args, kwargs)


class Constant(Expression):
    _instances = {}  # type: Dict[Type, Dict[Any, Any]]
    count = 0

    def __new__(cls, value: Any):
        try:
            if isinstance(value, Constant):
                value = value.value
            if type(value) not in Constant._instances:
                Constant._instances[type(value)] = {}
            instances = Constant._instances[type(value)]
            if value not in instances:
                instances[value] = super().__new__(cls)
            return instances[value]
        except TypeError as e:
            if "unhashable type" not in e.args[0]:
                raise e
        if id(value) not in Constant._instances[type(value)]:
            Constant._instances[type(value)][id(value)] = super().__new__(cls)
        return Constant._instances[type(value)][id(value)]

    def __getnewargs__(self):
        return (self._value,)

    def __init__(self, value: Any):
        try:
            if object.__getattribute__(self, "_exists"):
                return
        except:
            pass
        self._value = value
        self._exists = True
        self._id = Constant.count
        Constant.count += 1

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


class Symbol(Expression):
    _instances = {}  # type: Dict[Type[Symbol], Dict[str, Symbol]]
    _exists = False

    def __new__(cls, identifier: Union[Constant, str]):
        if isinstance(identifier, Constant):
            identifier = identifier.value
        if not isinstance(identifier, str):
            identifier = str(identifier)
        if cls not in Symbol._instances:
            Symbol._instances[cls] = {}
        if identifier not in Symbol._instances[cls]:
            Symbol._instances[cls][identifier] = super().__new__(cls)
        return Symbol._instances[cls][identifier]

    def __getnewargs__(self):
        return (self.identifier,)

    def __init__(self, identifier: Union[Constant, str]):
        if self._exists:
            return
        if isinstance(identifier, Constant):
            identifier = identifier.value
        if not isinstance(identifier, str):
            identifier = str(identifier)
        self.identifier = identifier
        self._value = None
        self._exists = True

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
    def __new__(cls, identifier: Union[Constant, str], type: Type, default: Any = None):
        return super().__new__(cls, f"P({identifier})")

    def __getnewargs__(self):
        return (self.identifier, self.type, self.default)

    def __init__(
        self, identifier: Union[Constant, str], type: Type, default: Any = None
    ):
        super().__init__(f"P({identifier})")
        self.name = str(identifier)
        self.type = type
        if isinstance(default, Constant):
            self.default = default.value
        else:
            self.default = default

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_concrete:
            return repr(self.value)
        return f"Parameter({self.name!r}, type={self.type.__name__}, default={self.default!r})"


class Network(Symbol):
    def __new__(cls, identifier: Union[Constant, str] = "N"):
        return super().__new__(cls, identifier)

    def __getnewargs__(self):
        return (self.identifier,)

    def __init__(self, identifier: Union[Constant, str] = "N"):
        super().__init__(identifier)

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


class Image(Expression):
    def __init__(self, path: Union[Expression, str]):
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
    ):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @property
    def value(self):
        if self.is_concrete:
            args = tuple(arg.value for arg in self.args)
            kwargs = {name: value.value for name, value in self.kwargs.items()}
            return self.function.value(*args, **kwargs)
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
    def __init__(self, *expr: Expression):
        super().__init__()
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
    def __init__(self, expr1: Expression, expr2: Expression, expr3: Expression):
        self.expr1 = expr1
        self.expr2 = expr2
        self.expr3 = expr3

    def __repr__(self):
        return f"{type(self).__name__}({self.expr1!r}, {self.expr2!r}, {self.expr3!r})"

    def __str__(self):
        return f"{type(self).__name__}({self.expr1}, {self.expr2}, {self.expr3})"


class BinaryExpression(Expression):
    def __init__(self, expr1: Expression, expr2: Expression):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return f"{type(self).__name__}({self.expr1!r}, {self.expr2!r})"

    def __str__(self):
        return f"({self.expr1} {self.OPERATOR} {self.expr2})"


class UnaryExpression(Expression):
    def __init__(self, expr: Expression):
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
    def __init__(self, start: Any, stop: Any, step: Any):
        start = start if isinstance(start, Expression) else Constant(start)
        stop = stop if isinstance(stop, Expression) else Constant(stop)
        step = step if isinstance(step, Expression) else Constant(step)
        super().__init__(start, stop, step)

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
    ):
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


def __argcmp_helper(cmp_fn, A, Ai, i, j) -> Union[Constant, IfThenElse]:
    if j == len(Ai):
        return Constant(i)
    return IfThenElse(
        cmp_fn(A[Ai[i]], A[Ai[j]]),
        __argcmp_helper(cmp_fn, A, Ai, i, j + 1),
        __argcmp_helper(cmp_fn, A, Ai, j, j + 1),
    )


def __argcmp(cmp_fn, A: Expression) -> Union[Constant, IfThenElse]:
    if not isinstance(A, Expression):
        return Constant(np.argmax(A))
    elif isinstance(A, FunctionCall) and isinstance(A.function, Network):
        if not A.function.is_concrete:
            raise RuntimeError(
                "argcmp can not be applied to outputs of non-concrete networks"
            )
        output_shape = A.function.value.output_shape[0]
        Ai = list(np.ndindex(output_shape))
        argcmp = __argcmp_helper(cmp_fn, A, Ai, 0, 1)
        return argcmp
    else:
        raise RuntimeError(f"Unsupported type for argcmp input: {type(A)}")


def __argmax(A: Expression) -> Union[Constant, IfThenElse]:
    return __argcmp(lambda a, b: a >= b, A)


def __argmin(A: Expression) -> Union[Constant, IfThenElse]:
    return __argcmp(lambda a, b: a <= b, A)


def __argcmp_eq(cmp_fn, A: Expression, c: Constant) -> Union[And, Constant]:
    if not isinstance(A, Expression):
        return Constant(np.argmax(A) == c)
    elif isinstance(A, FunctionCall) and isinstance(A.function, Network):
        if not A.function.is_concrete:
            raise RuntimeError(
                "argcmp can not be applied to outputs of non-concrete networks"
            )
        ci = c.value
        output_shape = A.function.value.output_shape[0]
        Ai = list(np.ndindex(output_shape))
        return And(*[cmp_fn(A[Ai[ci]], A[Ai[i]]) for i in range(len(Ai)) if i != ci])
    else:
        raise RuntimeError(f"Unsupported type for argcmp input: {type(A)}")


def __argcmp_neq(cmp_fn, A: Expression, c: Constant) -> Union[Or, Constant]:
    if not isinstance(A, Expression):
        return Constant(np.argmax(A) != c)
    elif isinstance(A, FunctionCall) and isinstance(A.function, Network):
        if not A.function.is_concrete:
            raise RuntimeError(
                "argcmp can not be applied to outputs of non-concrete networks"
            )
        ci = c.value
        output_shape = A.function.value.output_shape[0]
        Ai = list(np.ndindex(output_shape))
        return Or(*[~cmp_fn(A[Ai[ci]], A[Ai[i]]) for i in range(len(Ai)) if i != ci])
    else:
        raise RuntimeError(f"Unsupported type for argcmp input: {type(A)}")


def __argmax_eq(A: FunctionCall, c: Constant) -> Union[And, Constant]:
    if len(A.args) > 1 or len(A.kwargs) != 0:
        raise RuntimeError("Too many arguments for argmax")
    return __argcmp_eq(lambda a, b: a >= b, A.args[0], c)


def __argmax_neq(A: FunctionCall, c: Constant) -> Union[Or, Constant]:
    if len(A.args) > 1 or len(A.kwargs) != 0:
        raise RuntimeError("Too many arguments for argmax")
    return __argcmp_neq(lambda a, b: a >= b, A.args[0], c)


def __argmin_eq(A: FunctionCall, c: Constant) -> Union[And, Constant]:
    if len(A.args) > 1 or len(A.kwargs) != 0:
        raise RuntimeError("Too many arguments for argmin")
    return __argcmp_eq(lambda a, b: a <= b, A.args[0], c)


def __argmin_neq(A: FunctionCall, c: Constant) -> Union[Or, Constant]:
    if len(A.args) > 1 or len(A.kwargs) != 0:
        raise RuntimeError("Too many arguments for argmin")
    return __argcmp_neq(lambda a, b: a <= b, A.args[0], c)


def __abs(x) -> Union[Constant, IfThenElse]:
    if not isinstance(x, Expression):
        return Constant(np.abs(x))
    return IfThenElse(x >= 0, x, -x)


_BUILTIN_FUNCTION_TRANSFORMS = {
    (np.argmax, Equal): __argmax_eq,
    (np.argmin, Equal): __argmin_eq,
    (np.argmax, NotEqual): __argmax_neq,
    (np.argmin, NotEqual): __argmin_neq,
}  # type: Dict[Any, Callable[[FunctionCall, Constant], Expression]]

_BUILTIN_FUNCTIONS = {
    abs: __abs,
    np.abs: __abs,
    np.argmax: __argmax,
    np.argmin: __argmin,
}  # type: Dict[Any, Callable[..., Expression]]

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

