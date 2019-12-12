import ast

from pathlib import Path

from . import base
from .base import *


class PropertyParserError(Exception):
    pass


class Py2PropertyTransformer(ast.NodeTransformer):
    def __init__(self):
        self.universal_variables = {}
        self.existential_variables = {}

    def visit_Call(self, node: ast.Call):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if isinstance(node.func, ast.Name) and node.func.id in ["Forall", "Exists"]:
            if len(node.args) != 2:
                raise ValueError("%s takes 2 arguments." % node.func.id)
            if not isinstance(node.args[0], ast.Name):
                raise PropertyParserError(
                    "The first argument to %s must be a variable name." % node.func.id
                )
            symbol_name = node.args[0]
            sym_func = ast.Name("Symbol", ast.Load(), **attributes)
            args = [ast.Str(symbol_name.id, **attributes)]  # type: List[ast.AST]
            variable = ast.Call(sym_func, args, [], **attributes)

            orig_expr = super().visit(node.args[1])
            lambda_args = ast.arguments(
                [ast.arg(symbol_name.id, None, **attributes)], None, [], [], None, []
            )
            lambda_func = ast.Lambda(lambda_args, orig_expr, **attributes)
            new_expr = ast.Call(lambda_func, [variable], [], **attributes)

            new_args = [variable, lambda_func]
            new_node = ast.Call(node.func, new_args, [], **attributes)

            return new_node
        elif isinstance(node.func, ast.Name) and node.func.id == "Parameter":
            return super().generic_visit(node)
        elif isinstance(node.func, ast.Name):
            value = globals().get(node.func.id, None)
            if (
                value is not None
                and isinstance(value, type)
                and issubclass(value, Expression)
            ):
                return super().generic_visit(node)
            make_func = ast.Name("make_function", ast.Load(), **attributes)
            args = [node.func]
            func_expr = ast.Call(make_func, args, [], **attributes)

            new_node = ast.Call(func_expr, node.args, node.keywords, **attributes)
            return new_node
        elif isinstance(node.func, ast.Attribute):
            make_func = ast.Name("make_function", ast.Load(), **attributes)
            args = [node.func]
            func_expr = ast.Call(make_func, args, [], **attributes)

            new_node = ast.Call(func_expr, node.args, node.keywords, **attributes)
            return new_node
        elif isinstance(node.func, ast.Subscript):
            # ignore for now
            pass
        return super().generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if len(node.ops) == 1:
            return super().generic_visit(node)
        comparisons = []
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, ast.LtE):
                func = ast.Name("LessThanOrEqual", ast.Load(), **attributes)
            elif isinstance(op, ast.Lt):
                func = ast.Name("LessThan", ast.Load(), **attributes)
            elif isinstance(op, ast.GtE):
                func = ast.Name("GreaterThanOrEqual", ast.Load(), **attributes)
            elif isinstance(op, ast.Gt):
                func = ast.Name("GreaterThan", ast.Load(), **attributes)
            else:
                raise ValueError("Unknown comparison function: %s" % op)
            comparisons.append(ast.Call(func, [left, right], [], **attributes))
            left = right
        and_func = ast.Name("And", ast.Load(), **attributes)
        new_node = ast.Call(and_func, comparisons, [], **attributes)
        return new_node


class LimitQuantifiers(ExpressionVisitor):
    def __init__(self):
        super().__init__()
        self.at_top_level = True
        self.top_level_quantifier = None

    def __call__(self, phi):
        self.at_top_level = True
        self.top_level_quantifier = None
        if isinstance(phi, Quantifier):
            self.top_level_quantifier = phi.__class__
        self.visit(phi)

    def generic_visit(self, expr):
        self.at_top_level = False
        super().generic_visit(expr)

    def visit_Exists(self, expr):
        if not self.at_top_level:
            raise PropertyParserError("Quantifiers are only allowed at the top level.")
        if not isinstance(expr, self.top_level_quantifier):
            raise PropertyParserError(
                "Quantifiers at the top level must be of the same type."
            )
        self.visit(expr.expression)

    def visit_Forall(self, expr):
        if not self.at_top_level:
            raise PropertyParserError("Quantifiers are only allowed at the top level.")
        if not isinstance(expr, self.top_level_quantifier):
            raise PropertyParserError(
                "Quantifiers at the top level must be of the same type."
            )
        self.visit(expr.expression)


class SymbolFactory(dict):
    def __getitem__(self, item):
        if item not in self:
            assert isinstance(item, str)
            super().__setitem__(item, Symbol(item))
        return super().__getitem__(item)


class ParameterParser:
    def __init__(self, args: Optional[List[str]]):
        self.args = args

    def parse(self, name: str, type, default=None) -> Constant:
        import argparse

        if default is not None and not isinstance(default, type):
            raise PropertyParserError(
                f"Default parameter value, {default}, is not of specified type {type.__name__}."
            )

        parser = argparse.ArgumentParser()
        parser.add_argument(f"--prop.{name}", type=type, default=default)
        known_args, unknown_args = parser.parse_known_args(self.args)

        parameter_value = getattr(known_args, f"prop.{name}")
        if self.args is not None:
            self.args.clear()
            self.args.extend(unknown_args)

        if parameter_value is None:
            raise PropertyParserError(
                f"No argument was provided for parameter '{name}'. "
                f"Try adding a command line argument '--prop.{name}'."
            )
        return Constant(parameter_value)


def parse(path: Path, args: Optional[List[str]] = None):
    with open(path, "r") as f:
        module = ast.parse(f.read())
    for node in module.body[:-1]:
        if not isinstance(node, (ast.Assign, ast.Import, ast.ImportFrom)):
            raise PropertyParserError(node)
    property_node = module.body[-1]
    if not isinstance(property_node, ast.Expr):
        raise PropertyParserError()

    module = Py2PropertyTransformer().visit(module)

    attributes = {
        "lineno": property_node.lineno,
        "col_offset": property_node.col_offset,
    }
    module.body[-1] = ast.Assign(
        [ast.Name("phi", ast.Store(), **attributes)], property_node.value, **attributes
    )

    global_dict = SymbolFactory()
    global_dict.update(globals()["__builtins__"])
    global_dict.update(base.__dict__)

    parameter_parser = ParameterParser(args)
    global_dict.update(Parameter=parameter_parser.parse)

    code = compile(module, filename=path.name, mode="exec")
    exec(code, global_dict)
    phi = global_dict["phi"]
    LimitQuantifiers()(phi)

    return phi
