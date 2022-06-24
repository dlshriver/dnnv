from __future__ import annotations

from typing import List, Optional

from .. import expressions
from ..visitors import ExpressionVisitor
from .errors import ParserError


class LimitQuantifiers(ExpressionVisitor):
    def __init__(self):
        super().__init__()
        self.at_top_level = True
        self.top_level_quantifier = None

    def __call__(self, phi):
        self.at_top_level = True
        self.top_level_quantifier = None
        if isinstance(phi, expressions.Quantifier):
            self.top_level_quantifier = phi.__class__
        self.visit(phi)

    def generic_visit(self, expr):
        self.at_top_level = False
        super().generic_visit(expr)

    def visit_Exists(self, expr):
        if not self.at_top_level:
            raise ParserError("Quantifiers are only allowed at the top level")
        if not isinstance(expr, self.top_level_quantifier):
            raise ParserError("Quantifiers at the top level must be of the same type")
        self.visit(expr.expression)

    def visit_Forall(self, expr):
        if not self.at_top_level:
            raise ParserError("Quantifiers are only allowed at the top level")
        if not isinstance(expr, self.top_level_quantifier):
            raise ParserError("Quantifiers at the top level must be of the same type")
        self.visit(expr.expression)


def parse_cli(
    phi: expressions.Expression, args: Optional[List[str]] = None
) -> expressions.Expression:
    import argparse

    parser = argparse.ArgumentParser()

    parameters = set(
        expr for expr in phi.iter() if isinstance(expr, expressions.Parameter)
    )
    for parameter in parameters:
        parser.add_argument(
            f"--prop.{parameter.name}",
            type=parameter.type,
            default=parameter.default,
        )
    known_args, unknown_args = parser.parse_known_args(args)
    if args is not None:
        args.clear()
        args.extend(unknown_args)
    for parameter in parameters:
        parameter_value = getattr(known_args, f"prop.{parameter.name}")
        if isinstance(parameter_value, expressions.Expression):
            if not parameter_value.is_concrete:
                raise ParserError(
                    f"Parameter with non-concrete value: {parameter.name}"
                )
            parameter_value = parameter_value.value
        if parameter_value is None:
            raise ParserError(
                f"No argument was provided for parameter '{parameter.name}'. "
                f"Try adding a command line argument '--prop.{parameter.name}'."
            )
        parameter.concretize(parameter_value)
    return phi


__all__ = ["LimitQuantifiers", "parse_cli"]
