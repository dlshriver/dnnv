import unittest

from dnnv.properties import *
from dnnv.properties.context import get_context


class ToCNFTests(unittest.TestCase):
    def reset_property_context(self):
        get_context().reset()

    def setUp(self):
        self.reset_property_context()

    def test_and(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = a & b & c
        cnf_expr = expr.to_cnf()
        self.assertEqual(repr(expr), "And(Symbol('a'), Symbol('b'), Symbol('c'))")
        self.assertEqual(repr(cnf_expr), "And(Symbol('a'), Symbol('b'), Symbol('c'))")

        expr = a & ~b & c
        cnf_expr = expr.to_cnf()
        self.assertEqual(repr(expr), "And(Not(Symbol('b')), Symbol('a'), Symbol('c'))")
        self.assertEqual(
            repr(cnf_expr), "And(Not(Symbol('b')), Symbol('a'), Symbol('c'))"
        )

    def test_or(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = a | b | c
        cnf_expr = expr.to_cnf()
        self.assertEqual(repr(expr), "Or(Symbol('a'), Symbol('b'), Symbol('c'))")
        self.assertEqual(
            repr(cnf_expr), "And(Or(Symbol('a'), Symbol('b'), Symbol('c')))"
        )

        expr = a | ~b | c
        cnf_expr = expr.to_cnf()
        self.assertEqual(repr(expr), "Or(Not(Symbol('b')), Symbol('a'), Symbol('c'))")
        self.assertEqual(
            repr(cnf_expr), "And(Or(Not(Symbol('b')), Symbol('a'), Symbol('c')))"
        )

    def test_from_cnf(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = (a | b | c) & (~a | b)
        cnf_expr = expr.to_cnf()
        self.assertEqual(
            repr(expr),
            "And(Or(Not(Symbol('a')), Symbol('b')), "
            "Or(Symbol('a'), Symbol('b'), Symbol('c')))",
        )
        self.assertEqual(
            repr(cnf_expr),
            "And(Or(Not(Symbol('a')), Symbol('b')), "
            "Or(Symbol('a'), Symbol('b'), Symbol('c')))",
        )

        expr = (a | ~b | c) & (~a | b | ~c)
        cnf_expr = expr.to_cnf()
        self.assertEqual(
            repr(expr),
            "And("
            "Or(Not(Symbol('a')), Not(Symbol('c')), Symbol('b')), "
            "Or(Not(Symbol('b')), Symbol('a'), Symbol('c')))",
        )
        self.assertEqual(
            repr(cnf_expr),
            "And("
            "Or(Not(Symbol('a')), Not(Symbol('c')), Symbol('b')), "
            "Or(Not(Symbol('b')), Symbol('a'), Symbol('c')))",
        )

    def test_from_dnf(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = (a & b & ~c) | (~a & b)
        cnf_expr = expr.to_cnf()
        self.assertEqual(
            repr(expr),
            "Or(And(Not(Symbol('a')), Symbol('b')), And(Not(Symbol('c')), Symbol('a'), Symbol('b')))",
        )
        self.assertEqual(
            repr(cnf_expr),
            "And("
            "Or(Not(Symbol('a')), Not(Symbol('c'))), "
            "Or(Not(Symbol('a')), Symbol('a')), "
            "Or(Not(Symbol('a')), Symbol('b')), "
            "Or(Not(Symbol('c')), Symbol('b')), "
            "Or(Symbol('a'), Symbol('b')), "
            "Or(Symbol('b')))",
        )

        expr = (a & ~b) | (~a & b & ~c)
        cnf_expr = expr.to_cnf()
        self.assertEqual(
            repr(expr),
            "Or(And(Not(Symbol('a')), Not(Symbol('c')), Symbol('b')), And(Not(Symbol('b')), Symbol('a')))",
        )
        self.assertEqual(
            repr(cnf_expr),
            "And("
            "Or(Not(Symbol('a')), Not(Symbol('b'))), "
            "Or(Not(Symbol('a')), Symbol('a')), "
            "Or(Not(Symbol('b')), Not(Symbol('c'))), "
            "Or(Not(Symbol('b')), Symbol('b')), "
            "Or(Not(Symbol('c')), Symbol('a')), "
            "Or(Symbol('a'), Symbol('b')))",
        )

    def test_qf(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = ((Implies(a, (b & ~c)) & Implies(~a, (~b & ~c))) & (c | a)) | (
            a & ~b & ~c
        )
        cnf_expr = expr.to_cnf()
        self.assertEqual(
            repr(expr),
            "Or(And("
            "Implies(Not(Symbol('a')), And(Not(Symbol('b')), Not(Symbol('c')))), "
            "Implies(Symbol('a'), And(Not(Symbol('c')), Symbol('b'))), "
            "Or(Symbol('a'), Symbol('c'))), "
            "And(Not(Symbol('b')), Not(Symbol('c')), Symbol('a')))",
        )
        self.assertEqual(
            repr(cnf_expr),
            "And("
            "Or(Not(Symbol('a')), Not(Symbol('b')), Not(Symbol('c'))), "
            "Or(Not(Symbol('a')), Not(Symbol('b')), Symbol('b')), "
            "Or(Not(Symbol('a')), Not(Symbol('c')), Not(Symbol('c'))), "
            "Or(Not(Symbol('a')), Not(Symbol('c')), Symbol('a')), "
            "Or(Not(Symbol('a')), Not(Symbol('c')), Symbol('b')), "
            "Or(Not(Symbol('a')), Symbol('a'), Symbol('b')), "
            "Or(Not(Symbol('b')), Not(Symbol('b')), Symbol('a')), "
            "Or(Not(Symbol('b')), Not(Symbol('c')), Symbol('a')), "
            "Or(Not(Symbol('b')), Not(Symbol('c')), Symbol('a')), "
            "Or(Not(Symbol('b')), Symbol('a'), Symbol('a')), "
            "Or(Not(Symbol('b')), Symbol('a'), Symbol('c')), "
            "Or(Not(Symbol('c')), Not(Symbol('c')), Symbol('a')), "
            "Or(Not(Symbol('c')), Symbol('a'), Symbol('a')), "
            "Or(Not(Symbol('c')), Symbol('a'), Symbol('c')), "
            "Or(Symbol('a'), Symbol('a'), Symbol('c')))",
        )

    def test_forall(self):
        x = Symbol("x")

        expr = Forall(x, Implies((x > 0) & (x < 10), 2 * x < 20))
        cnf_expr = expr.to_cnf()
        self.assertEqual(
            repr(expr),
            "Forall("
            "Symbol('x'), "
            "Implies("
            "And(GreaterThan(Symbol('x'), 0), LessThan(Symbol('x'), 10)), "
            "LessThan(Multiply(2, Symbol('x')), 20)))",
        )
        self.assertEqual(
            repr(cnf_expr),
            "And(Or("
            "GreaterThanOrEqual(Symbol('x'), 10), "
            "LessThan(Multiply(2, Symbol('x')), 20), "
            "LessThanOrEqual(Symbol('x'), 0)))",
        )


if __name__ == "__main__":
    unittest.main()
