import ast
import unittest

from dnnv.properties.dsl import (
    parse,
    parse_cli,
    PropertyParserError,
    Py2PropertyTransformer,
    SymbolFactory,
)


class DSLTests(unittest.TestCase):
    def test_unsupported(self):
        unsupported_statements = [
            "x = (i for i in range(10))",  # GeneratorExp
            "x = [i for i in range(10)]",  # ListComp
            "x = {i for i in range(10)}",  # SetComp
            "x = {i: i for i in range(10)}",  # DictComp
            "x = min(*[0, 1])",  # Starred
            "y = await call()",  # Await
            "y = yield 95",  # Yield
            "yield from [0, 1, 2]",  # YieldFrom
            "t = (X, Y, Z)",  # Tuple with non-primitive types
            "l = [X, Y, Z]",  # List with non-primitive types
            "s = {X, Y, Z}",  # Set with non-primitive types
            "d = {X: 0, 'a': 2}",  # Dict with non-primitive key
            "d = {0: A, 'a': B}",  # Dict with non-primitive value
        ]
        for statement in unsupported_statements:
            node = ast.parse(statement)
            with self.assertRaises(PropertyParserError):
                node = Py2PropertyTransformer().visit(node)

    def test_Assign(self):
        node = ast.parse("test = 42")
        prop_parser = Py2PropertyTransformer()
        node = prop_parser.visit(node)
        self.assertIsInstance(node, ast.Module)
        self.assertIsInstance(node.body[0], ast.Assign)
        self.assertIsInstance(node.body[0].targets[0], ast.Name)

        node = ast.parse("test = lambda: None")
        prop_parser = Py2PropertyTransformer()
        node = prop_parser.visit(node)
        self.assertIn("test", prop_parser._lambda_aliases)

        node = ast.parse("arr[0] = 0.0")
        with self.assertRaises(PropertyParserError):
            node = Py2PropertyTransformer().visit(node)


if __name__ == "__main__":
    unittest.main()
