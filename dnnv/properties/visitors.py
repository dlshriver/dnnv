from .base import *


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
