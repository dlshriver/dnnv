import unittest

from dnnv.properties import *


class ToCNFTests(unittest.TestCase):
    def reset_property_context(self):
        # TODO : refactor property implementation so this can be removed
        # required to ensure concretized symbols don't carry over
        Constant._instances = {}
        Constant.count = 0
        Symbol._instances = {}

    def setUp(self):
        self.reset_property_context()

    def test_and(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = a & b & c
        cnf_expr = expr.to_cnf()
        self.assertEqual(str(expr), "(a & b & c)")
        self.assertEqual(str(cnf_expr), "(a & b & c)")

        expr = a & ~b & c
        cnf_expr = expr.to_cnf()
        self.assertEqual(str(expr), "(a & ~b & c)")
        self.assertEqual(str(cnf_expr), "(a & ~b & c)")

    def test_or(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = a | b | c
        cnf_expr = expr.to_cnf()
        self.assertEqual(str(expr), "(a | b | c)")
        self.assertEqual(str(cnf_expr), "((a | b | c))")

        expr = a | ~b | c
        cnf_expr = expr.to_cnf()
        self.assertEqual(str(expr), "(a | ~b | c)")
        self.assertEqual(str(cnf_expr), "((a | ~b | c))")

    def test_from_cnf(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = (a | b | c) & (~a | b)
        cnf_expr = expr.to_cnf()
        self.assertEqual(str(expr), "((a | b | c) & (~a | b))")
        self.assertEqual(str(cnf_expr), "((a | b | c) & (~a | b))")

        expr = (a | ~b | c) & (~a | b | ~c)
        cnf_expr = expr.to_cnf()
        self.assertEqual(str(expr), "((a | ~b | c) & (~a | b | ~c))")
        self.assertEqual(str(cnf_expr), "((a | ~b | c) & (~a | b | ~c))")

    def test_from_dnf(self):
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        expr = (a & b & ~c) | (~a & b)
        cnf_expr = expr.to_cnf()
        self.assertEqual(str(expr), "((a & b & ~c) | (~a & b))")
        self.assertEqual(
            str(cnf_expr),
            "((a | ~a) & (a | b) & (b | ~a) & (b | b) & (~c | ~a) & (~c | b))",
        )

        expr = (a & ~b) | (~a & b & ~c)
        cnf_expr = expr.to_cnf()
        self.assertEqual(str(expr), "((a & ~b) | (~a & b & ~c))")
        self.assertEqual(
            str(cnf_expr),
            "((a | ~a) & (a | b) & (a | ~c) & (~b | ~a) & (~b | b) & (~b | ~c))",
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
            str(expr),
            "(((a ==> (b & ~c)) & (~a ==> (~b & ~c)) & (c | a)) | (a & ~b & ~c))",
        )
        self.assertEqual(
            str(cnf_expr),
            "((b | ~a | a) & (b | ~a | ~b) & (b | ~a | ~c) "
            "& (~c | ~a | a) & (~c | ~a | ~b) & (~c | ~a | ~c) "
            "& (~b | a | a) & (~b | a | ~b) & (~b | a | ~c) "
            "& (~c | a | a) & (~c | a | ~b) & (~c | a | ~c) "
            "& (c | a | a) & (c | a | ~b) & (c | a | ~c))",
        )

    def test_forall(self):
        x = Symbol("x")

        expr = Forall(x, Implies((x > 0) & (x < 10), 2 * x < 20))
        cnf_expr = expr.to_cnf()
        self.assertEqual(
            str(expr), "Forall(x, (((x > 0) & (x < 10)) ==> ((2 * x) < 20)))"
        )
        self.assertEqual(str(cnf_expr), "(((x <= 0) | (x >= 10) | ((2 * x) < 20)))")


if __name__ == "__main__":
    unittest.main()
