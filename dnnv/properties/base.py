import numpy as np
import types

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ExpressionVisitor:
    def visit(self, expression):
        method_name = "visit_%s" % expression.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(expression)

    def generic_visit(self, expression):
        for value in expression.__dict__.values():
            if isinstance(value, Expression):
                self.visit(value)
            elif isinstance(value, (list, tuple, set)):
                for sub_value in value:
                    if isinstance(sub_value, Expression):
                        self.visit(sub_value)
        return expression


def find_expressions(phi, expr_class):
    exprs = set()
    for value in phi.__dict__.values():
        if isinstance(value, expr_class):
            exprs.add(value)
        if isinstance(value, Expression):
            exprs = exprs.union(find_expressions(value, expr_class))
        elif isinstance(value, (list, tuple, set)):
            for sub_value in value:
                if isinstance(sub_value, expr_class):
                    exprs.add(sub_value)
                if isinstance(sub_value, Expression):
                    exprs = exprs.union(find_expressions(sub_value, expr_class))
    return exprs


class Expression(ABC):
    def __init__(self):
        self.is_network_input = False
        self.is_network_output = False

    @property
    def networks(self):
        return list(find_expressions(self, Network))

    @property
    def variables(self):
        return [s for s in find_expressions(self, Symbol) if not s.is_concrete]

    def propagate_constants(self):
        from .transformers import PropagateConstants

        return PropagateConstants().visit(self)

    def to_cnf(self):
        from .transformers import ToCNF

        expr = ToCNF().visit(self)
        if isinstance(expr, Or):
            return And(expr)
        return expr


# Logical Operations


class LogicalExpression(Expression):
    @abstractmethod
    def __invert__(self):
        raise NotImplementedError()

    def __and__(self, other):
        return And(self, other)

    def __rand__(self, other):
        return And(other, self)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return Or(other, self)


class Not(LogicalExpression):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def __str__(self):
        return f"~{self.expr}"

    def __invert__(self):
        return self.expr


class BinaryLogicalExpression(LogicalExpression):
    def __init__(self, expr1, expr2):
        super().__init__()
        self.expr1 = expr1 if isinstance(expr1, Expression) else Constant(expr1)
        self.expr2 = expr2 if isinstance(expr2, Expression) else Constant(expr2)

    def __str__(self):
        return f"({self.expr1} {self.OPERATOR} {self.expr2})"


class Equal(BinaryLogicalExpression):
    OPERATOR = "=="

    def __invert__(self):
        return NotEqual(self.expr1, self.expr2)


class NotEqual(BinaryLogicalExpression):
    OPERATOR = "!="

    def __invert__(self):
        return Equal(self.expr1, self.expr2)


class LessThan(BinaryLogicalExpression):
    OPERATOR = "<"

    def __invert__(self):
        return GreaterThanOrEqual(self.expr1, self.expr2)


class LessThanOrEqual(BinaryLogicalExpression):
    OPERATOR = "<="

    def __invert__(self):
        return GreaterThan(self.expr1, self.expr2)


class GreaterThan(BinaryLogicalExpression):
    OPERATOR = ">"

    def __invert__(self):
        return LessThanOrEqual(self.expr1, self.expr2)


class GreaterThanOrEqual(BinaryLogicalExpression):
    OPERATOR = ">="

    def __invert__(self):
        return LessThan(self.expr1, self.expr2)


class AssociativeLogicalExpression(LogicalExpression):
    def __init__(self, *expr):
        super().__init__()
        self.expressions = []
        for expression in expr:
            if isinstance(expression, self.__class__):
                self.expressions.extend(expression.expressions)
            else:
                self.expressions.append(expression)
        self.expressions = list(set(self.expressions))

    def __str__(self):
        result_str = f" {self.OPERATOR} ".join(str(expr) for expr in self.expressions)
        return f"({result_str})"

    def __iter__(self):
        for expr in self.expressions:
            yield expr


class Or(AssociativeLogicalExpression):
    OPERATOR = "|"

    def __invert__(self):
        return And(*[~expr for expr in self.expressions])


class And(AssociativeLogicalExpression):
    OPERATOR = "&"

    def __invert__(self):
        return Or(*[~expr for expr in self.expressions])


class Implies(BinaryLogicalExpression):
    OPERATOR = "==>"

    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)
        self.antecedent = antecedent
        self.consequent = consequent

    def __invert__(self):
        return And(self.antecedent, ~self.consequent)


class Quantifier(LogicalExpression):
    def __init__(self, variable, formula):
        super().__init__()
        self.variable = variable
        if callable(formula):
            formula = formula(variable)
        self.expression = formula


class Forall(Quantifier):
    def __str__(self):
        return f"Forall({self.variable}, {self.expression})"

    def __invert__(self):
        return Exists(self.variable, lambda _: ~self.expression)


class Exists(Quantifier):
    def __str__(self):
        return f"Exists({self.variable}, {self.expression})"

    def __invert__(self):
        return Forall(self.variable, lambda _: ~self.expression)


# Arithmetic Operations


class ArithmeticExpression(Expression):
    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Subtract(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __eq__(self, other):
        return Equal(self, other)

    def __ne__(self, other):
        return NotEqual(self, other)

    def __ge__(self, other):
        return GreaterThanOrEqual(self, other)

    def __gt__(self, other):
        return GreaterThan(self, other)

    def __le__(self, other):
        return LessThanOrEqual(self, other)

    def __lt__(self, other):
        return LessThan(self, other)


class BinaryArithmeticExpression(ArithmeticExpression):
    def __init__(self, expr1, expr2):
        super().__init__()
        self.expr1 = expr1
        self.expr2 = expr2

    def __str__(self):
        return f"({self.expr1} {self.OPERATOR} {self.expr2})"


class Add(BinaryArithmeticExpression):
    OPERATOR = "+"


class Subtract(BinaryArithmeticExpression):
    OPERATOR = "-"


class Multiply(BinaryArithmeticExpression):
    OPERATOR = "*"


class Divide(BinaryArithmeticExpression):
    OPERATOR = "/"


# Symbols


class Symbol(ArithmeticExpression, LogicalExpression):
    _SSA = defaultdict(int)  # type: Dict[str, int]
    _Symbols = {}  # type: Dict[str, Symbol]

    def __init__(self, name):
        super().__init__()
        self.name = self._ssa(name)
        self._concrete_value = None

    @property
    def is_concrete(self):
        return self.concrete_value is not None

    @property
    def concrete_value(self):
        return self._concrete_value

    def concretize(self, value):
        self._concrete_value = value
        return self

    def _ssa(self, name):
        ssa_name = name
        if name in self._SSA:
            ssa_name = f"{name}_{self._SSA[name]}"
        self._SSA[name] += 1
        self._Symbols[ssa_name] = self
        return ssa_name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __invert__(self):
        return Not(self)

    def __enter__(self):
        return self

    def __exit__(self, f, exc_type, exc_value, traceback):
        pass


class Constant(Expression):
    def __init__(self, value):
        super().__init__()
        self._value = value
        if isinstance(value, Constant):
            self._value = value.value

    @property
    def value(self):
        return self._value

    def __hash__(self):
        value = self.value
        if isinstance(value, np.ndarray):
            return hash(value.tostring())
        return hash(self.value)

    def __str__(self):
        value = self.value
        if isinstance(value, np.ndarray):
            if np.product(value.shape) == 1:
                value = value.item()
                self._value = value
            else:
                value = f"ndarray({value.shape})"
        return f"{value}"

    def __invert__(self):
        if isinstance(self.value, bool):
            return Constant(not self.value)
        return Constant(~self.value)

    def __add__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value + other.value)
        if isinstance(other, Expression):
            return super().__add__(other)
        return Constant(self.value + other)

    def __radd__(self, other):
        if isinstance(other, Constant):
            return Constant(other.value + self.value)
        if isinstance(other, Expression):
            return super().__radd__(other)
        return Constant(other + self.value)

    def __sub__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value - other.value)
        if isinstance(other, Expression):
            return super().__sub__(other)
        return Constant(self.value - other)

    def __rsub__(self, other):
        if isinstance(other, Constant):
            return Constant(other.value - self.value)
        if isinstance(other, Expression):
            return super().__rsub__(other)
        return Constant(other - self.value)

    def __mul__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value * other.value)
        if isinstance(other, Expression):
            return super().__mul__(other)
        return Constant(self.value * other)

    def __rmul__(self, other):
        if isinstance(other, Constant):
            return Constant(other.value * self.value)
        if isinstance(other, Expression):
            return super().__rmul__(other)
        return Constant(other * self.value)

    def __truediv__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value / other.value)
        if isinstance(other, Expression):
            return super().__truediv__(other)
        return Constant(self.value / other)

    def __rtruediv__(self, other):
        if isinstance(other, Constant):
            return Constant(other.value / self.value)
        if isinstance(other, Expression):
            return super().__rtruediv__(other)
        return Constant(other / self.value)

    def __and__(self, other):
        if isinstance(other, bool):
            other = Constant(other)
        if not isinstance(other, Constant):
            return super().__and__(other)
        return Constant(self.value & other.value)

    def __or__(self, other):
        if isinstance(other, bool):
            other = Constant(other)
        if not isinstance(other, Constant):
            return super().__or__(other)
        return Constant(self.value | other.value)

    def __eq__(self, other):
        if isinstance(other, (float, int, bool, str)):
            other = Constant(other)
        if not isinstance(other, Constant):
            return super().__eq__(other)
        return Constant(self.value == other.value)

    def __ne__(self, other):
        if isinstance(other, (float, int, bool, str)):
            other = Constant(other)
        if not isinstance(other, Constant):
            return super().__ne__(other)
        return Constant(self.value != other.value)

    def __ge__(self, other):
        if isinstance(other, (float, int, bool, str)):
            other = Constant(other)
        if not isinstance(other, Constant):
            return super().__ge__(other)
        return Constant(self.value >= other.value)

    def __gt__(self, other):
        if isinstance(other, (float, int, bool, str)):
            other = Constant(other)
        if not isinstance(other, Constant):
            return super().__gt__(other)
        return Constant(self.value > other.value)

    def __le__(self, other):
        if isinstance(other, (float, int, bool, str)):
            other = Constant(other)
        if not isinstance(other, Constant):
            return super().__le__(other)
        return Constant(self.value <= other.value)

    def __lt__(self, other):
        if isinstance(other, (float, int, bool, str)):
            other = Constant(other)
        if not isinstance(other, Constant):
            return super().__lt__(other)
        return Constant(self.value < other.value)

    def __neg__(self):
        if not isinstance(self.value, (float, int, bool, np.ndarray)):
            return super().__neg__()
        return Constant(-self.value)


class Image(Constant):
    def __init__(self, path=None, shift=None, scale=None):
        super().__init__("Image")
        self.path = None
        if path is not None:
            if isinstance(path, Constant):
                self.path = Path(path.value)
            elif isinstance(path, str):
                self.path = Path(path)
            elif isinstance(path, Path):
                self.path = path
            else:
                raise ValueError("Path must be a path-like object.")
        self.shift = shift
        self.scale = scale

    @property
    def value(self):
        img = np.expand_dims(np.load(self.path), 0)
        img = img.astype(np.float32)
        if self.scale is not None:
            img = self.scale * img
        if self.shift is not None:
            img = img + self.shift
        return img

    def __str__(self):
        result_string = f'Image("{self.path}")'
        if self.shift is not None:
            result_string = f"{result_string}{self.shift:+g}"
        if self.scale is not None:
            result_string = f"{self.scale:g}*{result_string}"
        return result_string

    def __add__(self, value):
        new_shift = value
        if isinstance(value, Constant):
            new_shift = value.value
        if self.shift is not None:
            new_shift += self.shift
        return Image(self.path, new_shift, self.scale)

    def __sub__(self, value):
        if isinstance(value, Constant):
            value = value.value
        return self.__add__(-value)

    def __mul__(self, value):
        new_shift = self.shift
        new_scale = value
        if isinstance(value, Constant):
            new_scale = value.value
        if self.scale is not None:
            new_scale *= self.scale
        if self.shift is not None:
            new_shift *= value
        return Image(self.path, new_shift, new_scale)

    def __truediv__(self, value):
        if isinstance(value, Constant):
            value = value.value
        return self.__mul__(1 / value)


class Function(Symbol):
    def __call__(self, *args, **kwargs):
        return FunctionCall(self, args, kwargs)


class FunctionCall(ArithmeticExpression):
    def __init__(
        self,
        function,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        is_network_output: bool = False,
    ):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_network_output = is_network_output

    def __str__(self):
        args = ", ".join(str(arg) for arg in self.args)
        kwargs = ", ".join(f"{name}={value}" for name, value in self.kwargs.items())
        if args and kwargs:
            return f"{self.function}({args}, {kwargs})"
        elif args:
            return f"{self.function}({args})"
        elif kwargs:
            return f"{self.function}({kwargs})"
        return f"{self.function}()"


class Network(Function):
    def __init__(self, name="N"):
        super().__init__(name)

    def __call__(self, *x):
        for x_ in x:
            if isinstance(x_, Expression):
                x_.is_network_input = True
        result = super().__call__(*x)
        result.is_network_output = True
        return result

    def __getitem__(self, index):
        return SlicedNetwork.build(self, index)


class SlicedNetwork(Function):
    def __init__(self, name, network, slice_):
        super().__init__(name)
        self.network = network
        self.slice_ = slice_

    def __call__(self, *x):
        for x_ in x:
            if isinstance(x_, Expression):
                x_.is_network_input = True
        result = super().__call__(*x)
        result.is_network_output = True
        return result

    @property
    def concrete_value(self):
        if self._concrete_value is None and self.network._concrete_value is not None:
            self._concrete_value = self.network._concrete_value[self.slice_]
        return self._concrete_value

    def concretize(self, value):
        self.network.concretize(value)
        return self

    @classmethod
    def build(cls, network, index):
        if isinstance(index, slice):
            index = (index,)
        elif isinstance(index, int):
            index = (slice(None), index)
        if len(index) > 2:
            raise ValueError("Too many indices for network slicing.")
        slice_reprs = []
        if isinstance(index[0], slice):
            start = index[0].start
            stop = index[0].stop
            step = index[0].step
            if step is not None and step != 1:
                raise TypeError(
                    "Only step sizes of 1 are supported for network slices."
                )
            start_s = start if start else ""
            stop_s = stop if stop else ""
            slice_reprs.append(f"{start_s}:{stop_s}")
        else:
            raise TypeError("Unexpected type for dimension 0 of network slice.")
        if len(index) > 1:
            if isinstance(index[1], int):
                slice_reprs.append(str(index[1]))
            elif isinstance(index[1], slice):
                start = index[1].start
                stop = index[1].stop
                step = index[1].step
                if step is not None and step != 1:
                    raise TypeError(
                        "Only step sizes of 1 are supported for network slices."
                    )
                start_s = start if start else ""
                stop_s = stop if stop else ""
                slice_reprs.append(f"{start_s}:{stop_s}")
            else:
                raise TypeError("Unexpected type for dimension 1 of network slice.")
        slice_str = ",".join(slice_reprs)
        sliced_name = f"{network.name}[{slice_str}]"
        if sliced_name in network._Symbols:
            return network._Symbols[sliced_name]
        return SlicedNetwork(sliced_name, network, index)

    def __getitem__(self, index):
        return SlicedNetwork.build(self, index)


CONCRETE_FUNCS = {}  # type: Dict[int, Function]


def make_function(function):
    if isinstance(function, Function):
        return function
    elif isinstance(function, types.LambdaType) and function.__name__ == "<lambda>":
        name = f"{function.__module__}.{function.__name__}"
    elif isinstance(function, types.BuiltinFunctionType):
        name = f"{function.__name__}"
    elif isinstance(function, types.FunctionType):
        name = f"{function.__module__}.{function.__name__}"
    elif isinstance(function, np.ufunc):
        name = f"numpy.{function.__name__}"
    elif isinstance(function, type) and callable(function):
        name = f"{function.__module__}.{function.__name__}"
    else:
        raise ValueError("Unsupported function type: %s" % function)
    func_id = id(function)
    if func_id not in CONCRETE_FUNCS:
        function_expr = Function(name)
        function_expr.concretize(function)
        CONCRETE_FUNCS[func_id] = function_expr
    return CONCRETE_FUNCS[func_id]


argmax = np.argmax
argmin = np.argmin
