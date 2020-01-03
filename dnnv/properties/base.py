import numpy as np
import types

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


def find_symbols(phi: "Expression", symbol_class: Type["Symbol"] = "Symbol"):
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
    return symbols


class Expression:
    def concretize(self, **kwargs):
        symbols = {s.identifier: s for s in find_symbols(self)}
        for name, value in kwargs:
            if name not in symbols:
                raise ValueError(f"Unknown identifier: {name!r}")
            symbols[name].concretize(value)
        return self

    def propagate_constants(self):
        from dnnv.properties.transformers import PropagateConstants

        return PropagateConstants().visit(self)

    def to_cnf(self):
        from dnnv.properties.transformers import ToCNF

        return ToCNF().visit(self)

    @property
    def networks(self):
        return list(find_symbols(self, Network))

    @property
    def parameters(self):
        return list(find_symbols(self, Parameter))

    @property
    def variables(self):
        return [s for s in find_symbols(self) if not s.is_concrete]

    def __getattr__(self, name):
        if isinstance(name, Expression):
            return Attribute(self, name)
        return Attribute(self, Constant(name))

    def __getitem__(self, index) -> "Subscript":
        if isinstance(index, Expression):
            return Subscript(self, index)
        return Subscript(self, Constant(index))

    def __add__(self, other) -> "Add":
        if isinstance(other, Expression):
            return Add(self, other)
        return Add(self, Constant(other))

    def __sub__(self, other) -> "Subtract":
        if isinstance(other, Expression):
            return Subtract(self, other)
        return Subtract(self, Constant(other))

    def __mul__(self, other) -> "Multiply":
        if isinstance(other, Expression):
            return Multiply(self, other)
        return Multiply(self, Constant(other))

    def __truediv__(self, other) -> "Divide":
        if isinstance(other, Expression):
            return Divide(self, other)
        return Divide(self, Constant(other))

    def __eq__(self, other) -> "Equal":
        if isinstance(other, Expression):
            return Equal(self, other)
        return Equal(self, Constant(other))

    def __ne__(self, other) -> "NotEqual":
        if isinstance(other, Expression):
            return NotEqual(self, other)
        return NotEqual(self, Constant(other))

    def __ge__(self, other):
        if isinstance(other, Expression):
            return GreaterThanOrEqual(self, other)
        return GreaterThanOrEqual(self, Constant(other))

    def __gt__(self, other):
        if isinstance(other, Expression):
            return GreaterThan(self, other)
        return GreaterThan(self, Constant(other))

    def __le__(self, other):
        if isinstance(other, Expression):
            return LessThanOrEqual(self, other)
        return LessThanOrEqual(self, Constant(other))

    def __lt__(self, other):
        if isinstance(other, Expression):
            return LessThan(self, other)
        return LessThan(self, Constant(other))

    def __and__(self, other):
        return And(self, other)

    def __rand__(self, other):
        return And(other, self)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return Or(other, self)

    def __invert__(self):
        return Not(self)

    def __call__(self, *args: "Expression", **kwargs: "Expression"):
        return FunctionCall(self, args, kwargs)


class Symbol(Expression):
    _instances = {}  # type: Dict[Type[Symbol], Dict[str, Symbol]]

    def __new__(cls, identifier: str):
        if cls not in Symbol._instances:
            Symbol._instances[cls] = {}
        if identifier not in Symbol._instances[cls]:
            Symbol._instances[cls][identifier] = super().__new__(cls)
        return Symbol._instances[cls][identifier]

    def __init__(self, identifier: str):
        if not isinstance(getattr(self, "identifier"), Attribute):
            return
        self.identifier = identifier
        self._value = None

    @property
    def value(self):
        return self._value

    @property
    def is_concrete(self):
        return self._value is not None

    def concretize(self, value):
        self._value = value

    def __hash__(self):
        return hash(self.identifier)

    def __str__(self):
        return self.identifier


def _get_function_name(function: Callable) -> str:
    if isinstance(function, types.LambdaType) and function.__name__ == "<lambda>":
        return f"{function.__module__}.{function.__name__}"
    elif isinstance(function, types.BuiltinFunctionType):
        return f"{function.__name__}"
    elif isinstance(function, types.FunctionType):
        return f"{function.__module__}.{function.__name__}"
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


class Parameter(Symbol):
    def __new__(cls, identifier: str, *args, **kwargs):
        return super().__new__(cls, identifier)

    def __init__(self, identifier: str, type: Type, default: Any = None):
        super().__init__(identifier)
        self.type = type
        self.default = default

    def __str__(self):
        if self.is_concrete:
            return repr(self.value)
        return f"Parameter({self.identifier!r}, type={self.type.__name__}, default={self.default!r})"


class Network(Symbol):
    def __new__(cls, identifier: str = "N"):
        return super().__new__(cls, identifier)

    def __init__(self, identifier: str = "N"):
        super().__init__(identifier)


class Constant(Expression):
    def __init__(self, value: Any):
        self._value = value

    def __str__(self):
        if isinstance(self._value, np.ndarray):
            return "".join(
                np.array2string(
                    self._value,
                    max_line_width=np.inf,
                    precision=3,
                    threshold=5,
                    edgeitems=2,
                ).split("\n")
            ).replace("  ", " ")
        return str(self._value)

    @property
    def value(self):
        return self._value


class Image(Expression):
    def __init__(self, path: Expression):
        self.path = path

    @classmethod
    def load(cls, path: Expression):
        if not isinstance(path, Constant):
            return Image(path)
        # TODO : handle other image formats
        return Constant(np.load(path.value)[None, :].astype(np.float32))

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
        elif kwargs:
            return f"{function_name}({kwargs_str})"
        return f"{function_name}()"


class Attribute(Expression):
    def __init__(self, expr: Expression, name: Expression):
        self.expr = expr
        self.name = name

    def __str__(self):
        return f"{self.expr}.{self.name}"


class Subscript(Expression):
    def __init__(self, expr: Expression, index: Expression):
        self.expr = expr
        self.index = index

    def __str__(self):
        return f"{self.expr}[{self.index}]"


class AssociativeExpression(Expression):
    def __init__(self, *expr: Expression):
        super().__init__()
        self.expressions = []  # type: List[Expression]
        for expression in expr:
            if isinstance(expression, self.__class__):
                self.expressions.extend(expression.expressions)
            else:
                self.expressions.append(expression)

    def __str__(self):
        result_str = f" {self.OPERATOR} ".join(str(expr) for expr in self.expressions)
        return f"({result_str})"

    def __iter__(self):
        for expr in self.expressions:
            yield expr


class BinaryExpression(Expression):
    def __init__(self, expr1: Expression, expr2: Expression):
        self.expr1 = expr1
        self.expr2 = expr2

    def __str__(self):
        return f"({self.expr1} {self.OPERATOR} {self.expr2})"


class ArithmeticExpression(Expression):
    def __call__(self, *args: "Expression", **kwargs: "Expression"):
        raise ValueError("Arithmetic expressions are not callable.")


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


class Not(LogicalExpression):
    def __init__(self, expr: LogicalExpression):
        self.expr = expr

    def __str__(self):
        return f"~{self.expr}"

    def __invert__(self):
        return self.expr


class Equal(BinaryExpression, LogicalExpression):
    OPERATOR = "=="

    def __invert__(self):
        return NotEqual(self.expr1, self.expr2)


class NotEqual(BinaryExpression, LogicalExpression):
    OPERATOR = "!="

    def __invert__(self):
        return Equal(self.expr1, self.expr2)


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

    def __str__(self):
        return f"{self.__class__.__name__}({self.variable}, {self.expression})"


class Forall(Quantifier):
    def __invert__(self):
        return Exists(self.variable, lambda _: ~self.expression)


class Exists(Quantifier):
    def __invert__(self):
        return Forall(self.variable, lambda _: ~self.expression)


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
    "Forall",
    "Exists",
]

