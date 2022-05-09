from __future__ import annotations

import logging
import operator
from typing import Callable, Optional, Union

import numpy as np

from .....errors import DNNVError
from ....expressions import (
    And,
    ArithmeticExpression,
    Call,
    Constant,
    Expression,
    IfThenElse,
    LogicalExpression,
    Or,
)
from .base import FunctionSubstitutor


def get_arg(expr: Call):
    assert expr.function.is_concrete
    assert expr.function.value in Abs.__matches__
    assert len(expr.args) == 1
    assert len(expr.kwargs) == 0
    return expr.args[0]


class Abs(FunctionSubstitutor):
    __matches__ = {abs, np.abs, np.absolute, np.fabs}

    def __call__(self, f, *args, **kwargs) -> Union[Constant, IfThenElse]:
        assert len(args) == 1
        assert len(kwargs) == 0
        (x,) = args
        assert isinstance(x, ArithmeticExpression)
        if x.is_concrete:
            return Constant(abs(x.value))
        x_shape = x.ctx.shapes.get(x)
        if x_shape != ():
            # TODO : support this
            raise DNNVError(
                f"Unsupported shape for expression {x} in abs({x}): {x_shape}"
            )
        return IfThenElse(x >= 0, x, -x)

    @staticmethod
    def _substitute_cmp(
        cmp: Callable[[Expression, Expression], Expression],
        a: Expression,
        b: Expression,
        form: Optional[str],
    ):
        logger = logging.getLogger(__name__)
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value in Abs.__matches__
        ) and (
            isinstance(b, Call)
            and b.function.is_concrete
            and b.function.value in Abs.__matches__
        ):
            a = get_arg(a)
            b = get_arg(b)
            assert isinstance(a, ArithmeticExpression)
            assert isinstance(b, ArithmeticExpression)
            a_shape = a.ctx.shapes.get(a, None)
            b_shape = a.ctx.shapes.get(b, None)
            if a_shape is None or b_shape is None:
                return NotImplemented
            output_shape = np.broadcast_shapes(a_shape, b_shape)
            if np.all(np.asarray(output_shape) == 1):
                if form == "cnf":
                    return And(
                        Or(
                            a < 0, b < 0, cmp(a, b)
                        ),  # Implies(And(a >= 0, b>=0), cmp(a, b))
                        Or(
                            a >= 0, b < 0, cmp(-a, b)
                        ),  # Implies(And(a < 0, b>=0), cmp(-a, b))
                        Or(
                            a < 0, b >= 0, cmp(a, -b)
                        ),  # Implies(And(a >= 0, b<0), cmp(a, -b))
                        Or(
                            a >= 0, b >= 0, cmp(-a, -b)
                        ),  # Implies(And(a < 0, b<0), cmp(-a, -b))
                    )
                return Or(
                    And(a >= 0, b >= 0, cmp(a, b)),
                    And(a < 0, b >= 0, cmp(-a, b)),
                    And(a >= 0, b < 0, cmp(a, -b)),
                    And(a < 0, b < 0, cmp(-a, -b)),
                )
            if a_shape != output_shape:
                a = Constant(np.broadcast_to)(a, Constant(output_shape))
            if b_shape != output_shape:
                b = Constant(np.broadcast_to)(b, Constant(output_shape))
            if form == "cnf":
                return And(
                    *(
                        And(
                            Or(a[idx] < 0, b[idx] < 0, cmp(a[idx], b[idx])),
                            Or(a[idx] >= 0, b[idx] < 0, cmp(-a[idx], b[idx])),
                            Or(a[idx] < 0, b[idx] >= 0, cmp(a[idx], -b[idx])),
                            Or(a[idx] >= 0, b[idx] >= 0, cmp(-a[idx], -b[idx])),
                        )
                        for idx in np.ndindex(*output_shape)
                    )
                )
            # TODO : is there an efficient DNF form of this?
            logger.warning(
                "Expanding absolute value functions can significantly increase expression size!"
            )
            return And(
                *(
                    Or(
                        And(a[idx] >= 0, b[idx] >= 0, cmp(a[idx], b[idx])),
                        And(a[idx] < 0, b[idx] >= 0, cmp(-a[idx], b[idx])),
                        And(a[idx] >= 0, b[idx] < 0, cmp(a[idx], -b[idx])),
                        And(a[idx] < 0, b[idx] < 0, cmp(-a[idx], -b[idx])),
                    )
                    for idx in np.ndindex(*output_shape)
                )
            )
        elif (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value in Abs.__matches__
        ):
            a = get_arg(a)
            assert isinstance(a, ArithmeticExpression)
            a_shape = a.ctx.shapes.get(a, None)
            b_shape = a.ctx.shapes.get(b, None)
            if a_shape is None or b_shape is None:
                return NotImplemented
            output_shape = np.broadcast_shapes(a_shape, b_shape)
            if np.all(np.asarray(output_shape) == 1):
                if form == "cnf":
                    return And(
                        Or(b >= 0),
                        Or(a < 0, cmp(a, b)),  # Implies(a >= 0, cmp(a, b))
                        Or(a >= 0, cmp(-a, b)),  # Implies(a < 0, cmp(a, b))
                    )
                return Or(
                    And(b >= 0, a >= 0, cmp(a, b)),
                    And(b >= 0, a < 0, cmp(-a, b)),
                )
            a_ = a
            if a_shape != output_shape:
                a_ = Constant(np.broadcast_to)(a, Constant(output_shape))
            b_ = b
            if b_shape != output_shape:
                b_ = Constant(np.broadcast_to)(b, Constant(output_shape))
            if np.all(np.asarray(b_shape) == 1):
                b_gt_0: LogicalExpression = b >= 0
            else:
                b_gt_0 = And(*(b[b_idx] >= 0 for b_idx in np.ndindex(*b_shape)))
            if form == "cnf":
                return And(
                    b_gt_0,
                    *(
                        And(
                            Or(a_[idx] < 0, cmp(a_[idx], b_[idx])),
                            Or(a_[idx] >= 0, cmp(-a_[idx], b_[idx])),
                        )
                        for idx in np.ndindex(*output_shape)
                    ),
                )
            # TODO : is there an efficient DNF form of this?
            logger.warning(
                "Expanding absolute value functions can significantly increase expression size!"
            )
            return And(
                *(
                    Or(
                        And(b_[idx] >= 0, a_[idx] >= 0, cmp(a_[idx], b_[idx])),
                        And(b_[idx] >= 0, a_[idx] < 0, cmp(-a_[idx], b_[idx])),
                    )
                    for idx in np.ndindex(*output_shape)
                )
            )
        assert isinstance(b, Call)
        b = get_arg(b)
        assert isinstance(a, ArithmeticExpression)
        assert isinstance(b, ArithmeticExpression)
        a_shape = a.ctx.shapes.get(a, None)
        b_shape = a.ctx.shapes.get(b, None)
        if a_shape is None or b_shape is None:
            return NotImplemented
        output_shape = np.broadcast_shapes(a_shape, b_shape)
        if np.all(np.asarray(output_shape) == 1):
            if form == "cnf":
                return And(
                    Or(a >= 0),
                    Or(b < 0, cmp(a, b)),
                    Or(b >= 0, cmp(a, -b)),
                )
            return Or(
                And(a >= 0, b >= 0, cmp(a, b)),
                And(a >= 0, b < 0, cmp(a, -b)),
            )
        a_ = a
        if a_shape != output_shape:
            a_ = Constant(np.broadcast_to)(a, Constant(output_shape))
        b_ = b
        if b_shape != output_shape:
            b_ = Constant(np.broadcast_to)(b, Constant(output_shape))
        if np.all(np.asarray(a_shape) == 1):
            a_gt_0: LogicalExpression = a >= 0
        else:
            a_gt_0 = And(*(a[a_idx] >= 0 for a_idx in np.ndindex(*a_shape)))
        if form == "cnf":
            return And(
                a_gt_0,
                *(
                    And(
                        Or(b_[idx] < 0, cmp(a_[idx], b_[idx])),
                        Or(b_[idx] >= 0, cmp(a_[idx], -b_[idx])),
                    )
                    for idx in np.ndindex(*output_shape)
                ),
            )
        # TODO : is there an efficient DNF form of this?
        logger.warning(
            "Expanding absolute value functions can significantly increase expression size!"
        )
        return And(
            *(
                Or(
                    And(a[idx] >= 0, b[idx] >= 0, cmp(a[idx], b[idx])),
                    And(a[idx] >= 0, b[idx] < 0, cmp(a[idx], -b[idx])),
                )
                for idx in np.ndindex(*output_shape)
            )
        )

    @staticmethod
    def substitute_Equal(a: Expression, b: Expression, form=None) -> Expression:
        return Abs._substitute_cmp(operator.eq, a, b, form)

    @staticmethod
    def substitute_GreaterThan(a: Expression, b: Expression, form=None) -> Expression:
        return Abs._substitute_cmp(operator.gt, a, b, form)

    @staticmethod
    def substitute_GreaterThanOrEqual(
        a: Expression, b: Expression, form=None
    ) -> Expression:
        return Abs._substitute_cmp(operator.ge, a, b, form)

    @staticmethod
    def substitute_LessThan(a: Expression, b: Expression, form=None) -> Expression:
        return Abs._substitute_cmp(operator.lt, a, b, form)

    @staticmethod
    def substitute_LessThanOrEqual(
        a: Expression, b: Expression, form=None
    ) -> Expression:
        return Abs._substitute_cmp(operator.le, a, b, form)

    @staticmethod
    def substitute_NotEqual(a: Expression, b: Expression, form=None) -> Expression:
        return Abs._substitute_cmp(operator.ne, a, b, form)


__all__ = ["Abs"]
