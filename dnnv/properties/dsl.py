import ast
import types

from pathlib import Path
from typing import List, Optional

from . import expressions
from .visitors import ExpressionVisitor


class PropertyParserError(Exception):
    def __init__(self, msg: str, *args: object, lineno=None, col_offset=None) -> None:
        if lineno is not None:
            prefix = f"line {lineno}"
            if col_offset is not None:
                prefix = f"{prefix}, col {col_offset}"
            msg = f"{prefix}: {msg}"
        super().__init__(msg, *args)


class Py2PropertyTransformer(ast.NodeTransformer):
    def __init__(self):
        self._lambda_aliases = set()

    def visit_Assign(self, node: ast.Assign):
        if any(not isinstance(target, ast.Name) for target in node.targets):
            raise PropertyParserError(
                "Assigning to non-identifiers is not currently supported",
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        if isinstance(node.value, ast.Lambda):
            assert isinstance(node.targets[0], ast.Name)
            self._lambda_aliases.add(node.targets[0].id)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        kwargs = [self.visit(keyword) for keyword in node.keywords]

        if isinstance(func, ast.Name):
            value = expressions.__dict__.get(func.id, None)
            if (
                value is not None
                and isinstance(value, type)
                and issubclass(value, expressions.Expression)
            ):
                return ast.Call(func, args, kwargs, **attributes)
            if func.id in self._lambda_aliases:
                return ast.Call(func, args, kwargs, **attributes)
        call_name_node = ast.Name("Call", ast.Load(), **attributes)
        args_list_node = ast.Tuple(args, ast.Load(), **attributes)
        kwards_dict_node = ast.Dict(
            [ast.Str(kwarg.arg, **attributes) for kwarg in kwargs],
            [kwarg.value for kwarg in kwargs],
            **attributes,
        )
        new_node = ast.Call(
            call_name_node,
            [func, args_list_node, kwards_dict_node],
            [],
            **attributes,
        )
        new_node = ast.Call(
            ast.Attribute(new_node, "propagate_constants", ast.Load(), **attributes),
            [],
            [],
            **attributes,
        )
        return new_node

    def visit_Compare(self, node: ast.Compare):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if len(node.ops) == 1:
            return self.generic_visit(node)
        comparisons = []
        left = self.visit(node.left)
        for op, right in zip(node.ops, node.comparators):
            right = self.visit(right)
            if isinstance(op, ast.LtE):
                func = ast.Name("LessThanOrEqual", ast.Load(), **attributes)
            elif isinstance(op, ast.Lt):
                func = ast.Name("LessThan", ast.Load(), **attributes)
            elif isinstance(op, ast.GtE):
                func = ast.Name("GreaterThanOrEqual", ast.Load(), **attributes)
            elif isinstance(op, ast.Gt):
                func = ast.Name("GreaterThan", ast.Load(), **attributes)
            elif isinstance(op, ast.Eq):
                func = ast.Name("Equal", ast.Load(), **attributes)
            elif isinstance(op, ast.NotEq):
                func = ast.Name("NotEqual", ast.Load(), **attributes)
            else:
                raise PropertyParserError(
                    f"Unsupported comparison function: {op}",
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
            comparisons.append(ast.Call(func, [left, right], [], **attributes))
            left = right
        and_func = ast.Name("And", ast.Load(), **attributes)
        new_node = ast.Call(and_func, comparisons, [], **attributes)
        return new_node

    def visit_Constant(self, node: ast.Constant):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Ellipsis(self, node: ast.Ellipsis):  # pragma: no cover
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_NameConstant(self, node: ast.NameConstant):  # pragma: no cover
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Num(self, node: ast.Num):  # pragma: no cover
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Str(self, node: ast.Str):  # pragma: no cover
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def _ensure_primitive(self, expr):
        primitive_type = (
            ast.Constant,
            ast.Ellipsis,
            ast.NameConstant,
            ast.Num,
            ast.Str,
        )
        if isinstance(
            expr,
            primitive_type,
        ):
            return True
        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.operand, primitive_type):
            return True
        elif isinstance(expr, ast.Dict):
            for k in expr.keys:
                if not self._ensure_primitive(k):
                    return False
            for v in expr.values:
                if not self._ensure_primitive(v):
                    return False
            return True
        elif isinstance(expr, (ast.List, ast.Set, ast.Tuple)):
            for e in expr.elts:
                if not self._ensure_primitive(e):
                    return False
            return True
        return False

    def visit_Dict(self, node: ast.Dict):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if not self._ensure_primitive(node):
            raise PropertyParserError(
                "DNNP does not currently support definition of dicts containing non-primitive keys or values",
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_List(self, node: ast.List):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if not self._ensure_primitive(node):
            raise PropertyParserError(
                "DNNP does not currently support definition of lists containing non-primitive types",
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Set(self, node: ast.Set):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if not self._ensure_primitive(node):
            raise PropertyParserError(
                "DNNP does not currently support definition of sets containing non-primitive types",
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Tuple(self, node: ast.Tuple):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        if not self._ensure_primitive(node):
            raise PropertyParserError(
                "DNNP does not currently support definition of tuples containing non-primitive types",
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        const_func = ast.Name("Constant", ast.Load(), **attributes)
        return ast.Call(const_func, [node], [], **attributes)

    def visit_Slice(self, node: ast.Slice):
        attributes = {"lineno": 0, "col_offset": 0}

        start = (
            self.visit(node.lower)
            if node.lower is not None
            else ast.NameConstant(None, **attributes)
        )
        stop = (
            self.visit(node.upper)
            if node.upper is not None
            else ast.NameConstant(None, **attributes)
        )
        step = (
            self.visit(node.step)
            if node.step is not None
            else ast.NameConstant(None, **attributes)
        )

        slice_func = ast.Name("Slice", ast.Load(), **attributes)
        new_node = ast.Call(slice_func, [start, stop, step], [], **attributes)
        index_node = ast.Index(new_node)
        return index_node

    def visit_ExtSlice(self, node: ast.ExtSlice):
        dims = [
            dim
            for dim in node.dims
            if not isinstance(dim, (ast.Slice))
            and (
                not isinstance(dim, (ast.Index))
                or not self._ensure_primitive(dim.value)
            )
        ]
        if len(dims) > 0:
            raise PropertyParserError(
                "DNNP does not currently support definition of slices containing non-primitive types"
            )
        return self.generic_visit(node)

    def visit_Await(self, node: ast.Await):
        raise PropertyParserError(
            "DNNP does not support await expressions",
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def visit_Yield(self, node: ast.Yield):
        raise PropertyParserError(
            "DNNP does not support yield expressions",
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def visit_YieldFrom(self, node: ast.YieldFrom):
        raise PropertyParserError(
            "DNNP does not support yield from expressions",
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def visit_IfExp(self, node: ast.IfExp):
        attributes = {"lineno": node.lineno, "col_offset": node.col_offset}
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        ite_func = ast.Name("IfThenElse", ast.Load(), **attributes)
        new_node = ast.Call(ite_func, [test, body, orelse], [], **attributes)
        return new_node

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        raise PropertyParserError(
            "DNNP does not currently support generator expressions",
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def visit_DictComp(self, node: ast.DictComp):
        raise PropertyParserError(
            "DNNP does not currently support dict comprehensions",
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def visit_ListComp(self, node: ast.ListComp):
        raise PropertyParserError(
            "DNNP does not currently support list comprehensions",
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def visit_SetComp(self, node: ast.SetComp):
        raise PropertyParserError(
            "DNNP does not currently support set comprehensions",
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def visit_Starred(self, node: ast.Starred):
        raise PropertyParserError(
            "DNNP does not currently support starred expressions",
            lineno=node.lineno,
            col_offset=node.col_offset,
        )


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
            raise PropertyParserError("Quantifiers are only allowed at the top level")
        if not isinstance(expr, self.top_level_quantifier):
            raise PropertyParserError(
                "Quantifiers at the top level must be of the same type"
            )
        self.visit(expr.expression)

    def visit_Forall(self, expr):
        if not self.at_top_level:
            raise PropertyParserError("Quantifiers are only allowed at the top level")
        if not isinstance(expr, self.top_level_quantifier):
            raise PropertyParserError(
                "Quantifiers at the top level must be of the same type"
            )
        self.visit(expr.expression)


class SymbolFactory(dict):
    def __getitem__(self, item):
        if item not in self:
            assert isinstance(item, str)
            super().__setitem__(item, expressions.Symbol(item))
        result = super().__getitem__(item)
        if isinstance(result, types.LambdaType):
            return result
        if not isinstance(result, expressions.Expression) and (
            not isinstance(result, type)
            or not issubclass(result, expressions.Expression)
        ):
            result = expressions.Constant(result)
            super().__setitem__(item, result)
        return result


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
        if parameter_value is None:
            raise PropertyParserError(
                f"No argument was provided for parameter '{parameter.name}'. "
                f"Try adding a command line argument '--prop.{parameter.name}'."
            )
        parameter.concretize(parameter_value)
    return phi


def parse_ast(
    ast_node: ast.Module, path: Path = Path(), args: Optional[List[str]] = None
) -> expressions.Expression:
    for node in ast_node.body[:-1]:
        if not isinstance(node, (ast.Assign, ast.Import, ast.ImportFrom)) and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)
        ):
            raise PropertyParserError(
                f"Unsupported structure in property: {node}",
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
    property_node = ast_node.body[-1]
    if not isinstance(property_node, ast.Expr):
        raise PropertyParserError("No property expression found")

    module = Py2PropertyTransformer().visit(ast_node)

    attributes = {
        "lineno": property_node.lineno,
        "col_offset": property_node.col_offset,
    }
    module.body[-1] = ast.Assign(
        [ast.Name("phi", ast.Store(), **attributes)], property_node.value, **attributes
    )

    global_dict = SymbolFactory()
    global_dict.update(globals()["__builtins__"])
    base_dict = dict(expressions.__dict__)
    global_dict.update(base_dict)
    global_dict["__path__"] = path

    with expressions.Context():
        code = compile(module, filename=path.name, mode="exec")
        exec(code, global_dict)
        phi = global_dict["phi"]
        LimitQuantifiers()(phi)

        phi = phi.propagate_constants()
        phi = parse_cli(phi, args)

    return phi


def parse(path: Path, args: Optional[List[str]] = None) -> expressions.Expression:
    with open(path, "r") as f:
        module = ast.parse(f.read())
    return parse_ast(module, path, args)


__all__ = ["parse", "parse_ast", "PropertyParserError"]
