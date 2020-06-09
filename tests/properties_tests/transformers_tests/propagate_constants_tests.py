import unittest

from dnnv.properties import *
from dnnv.properties.context import get_context


class PropagateConstantsTests(unittest.TestCase):
    def reset_property_context(self):
        get_context().reset()

    def setUp(self):
        self.reset_property_context()

    def test_add_constants(self):
        expr = Add(Constant(3), Constant(4))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, 7)

        expr = Add(Constant(-100), Constant(41), Constant(39), Constant(15))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, -5)

    def test_and_constants(self):
        expr = And(Constant(True), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = And(Constant(True), Constant(False))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, False)

        expr = And(Constant(False), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, False)

        expr = And(Constant(False), Constant(False))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, False)

        expr = And(Constant(True), Constant(True), Constant(True), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = And(Constant(True), Constant(False), Constant(True), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, False)

        expr = And(False, Symbol("p"))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, False)

        expr = And(True, Symbol("p"))
        expr_ = expr.propagate_constants()
        self.assertIs(expr_, Symbol("p"))

    def test_divide_constants(self):
        expr = Divide(Constant(17), Constant(3))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertAlmostEqual(expr_.value, 17 / 3)

        expr = Divide(Constant(-10), Constant(2))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, -5)

        expr = Divide(Constant(8), Constant(0.1))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, 80)

    def test_multiply_constants(self):
        expr = Multiply(Constant(9), Constant(-5))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, -45)

        expr = Multiply(Constant(-10), Constant(0.1), Constant(3), Constant(-15))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, 45)

    def test_implies_constants(self):
        expr = Implies(Constant(True), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Implies(Constant(True), Constant(False))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, False)

        expr = Implies(Constant(False), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Implies(Constant(False), Constant(False))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Implies(False, Symbol("p"))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Implies(True, Symbol("p"))
        expr_ = expr.propagate_constants()
        self.assertIs(expr_, Symbol("p"))

    def test_or_constants(self):
        expr = Or(Constant(True), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Or(Constant(True), Constant(False))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Or(Constant(False), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Or(Constant(False), Constant(False))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, False)

        expr = Or(Constant(True), Constant(True), Constant(True), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Or(Constant(True), Constant(False), Constant(True), Constant(True))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Or(True, Symbol("p"))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, True)

        expr = Or(False, Symbol("p"))
        expr_ = expr.propagate_constants()
        self.assertIs(expr_, Symbol("p"))

    def test_subtract_constants(self):
        expr = Subtract(Constant(21), Constant(7))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, 14)

        expr = Subtract(Constant(-52), Constant(-9))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertEqual(expr_.value, -43)

        expr = Subtract(Constant(13), Constant(0.21))
        expr_ = expr.propagate_constants()
        self.assertIsInstance(expr_, Constant)
        self.assertAlmostEqual(expr_.value, 12.79)


if __name__ == "__main__":
    unittest.main()
