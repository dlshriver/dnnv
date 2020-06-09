import numpy as np
import unittest

from dnnv.properties.base import *
from dnnv.properties.context import get_context


class ExpressionTests(unittest.TestCase):
    def reset_property_context(self):
        get_context().reset()

    def setUp(self):
        self.reset_property_context()

    def test_expression(self):
        with self.assertRaises(TypeError):
            e = Expression()

    def test_value_expressions(self):
        e1 = Symbol("a") + Symbol("b")
        with self.assertRaises(ValueError):
            x = e1.value

        e1.concretize(a=1, b=2)
        self.assertEqual(e1.value, 3)

    def test_concretize_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        add_e1_e2 = e1 + e2
        self.assertIsInstance(add_e1_e2, Add)
        self.assertEqual(str(add_e1_e2), "(a + b)")

        add_e1_e2.concretize(a=11.2, b=-13.5)
        self.assertAlmostEqual(add_e1_e2.value, -2.3)

        with self.assertRaises(ValueError):
            add_e1_e2.concretize(c=0)

    def test_bool(self):
        e1 = Symbol("x") + Symbol("y")
        self.assertTrue(bool(e1))
        self.assertFalse(e1.concretize(x=0, y=0))
        self.assertTrue(e1.concretize(x=1, y=1))

    def test_add_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        add_e1_e2 = e1 + e2
        self.assertIsInstance(add_e1_e2, Add)
        self.assertEqual(str(add_e1_e2), "(a + b)")

        e3 = 5
        add_e3_e2 = e3 + e2
        self.assertIsInstance(add_e3_e2, Add)
        self.assertEqual(str(add_e3_e2), "(5 + b)")

        add_e1_e3 = e1 + e3
        self.assertIsInstance(add_e1_e3, Add)
        self.assertEqual(str(add_e1_e3), "(a + 5)")

        radd_e2_e1 = e2.__radd__(e1)
        self.assertIsInstance(radd_e2_e1, Add)
        self.assertEqual(str(radd_e2_e1), "(a + b)")

    def test_subtract_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        sub_e1_e2 = e1 - e2
        self.assertIsInstance(sub_e1_e2, Subtract)
        self.assertEqual(str(sub_e1_e2), "(a - b)")

        e3 = 5
        sub_e3_e2 = e3 - e2
        self.assertIsInstance(sub_e3_e2, Subtract)
        self.assertEqual(str(sub_e3_e2), "(5 - b)")

        sub_e1_e3 = e1 - e3
        self.assertIsInstance(sub_e1_e3, Subtract)
        self.assertEqual(str(sub_e1_e3), "(a - 5)")

        rsub_e2_e1 = e2.__rsub__(e1)
        self.assertIsInstance(rsub_e2_e1, Subtract)
        self.assertEqual(str(rsub_e2_e1), "(a - b)")

    def test_multiply_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        mul_e1_e2 = e1 * e2
        self.assertIsInstance(mul_e1_e2, Multiply)
        self.assertEqual(str(mul_e1_e2), "(a * b)")

        e3 = 5
        mul_e3_e2 = e3 * e2
        self.assertIsInstance(mul_e3_e2, Multiply)
        self.assertEqual(str(mul_e3_e2), "(5 * b)")

        mul_e1_e3 = e1 * e3
        self.assertIsInstance(mul_e1_e3, Multiply)
        self.assertEqual(str(mul_e1_e3), "(a * 5)")

        rmul_e2_e1 = e2.__rmul__(e1)
        self.assertIsInstance(rmul_e2_e1, Multiply)
        self.assertEqual(str(rmul_e2_e1), "(a * b)")

    def test_divide_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        div_e1_e2 = e1 / e2
        self.assertIsInstance(div_e1_e2, Divide)
        self.assertEqual(str(div_e1_e2), "(a / b)")

        e3 = 5
        div_e3_e2 = e3 / e2
        self.assertIsInstance(div_e3_e2, Divide)
        self.assertEqual(str(div_e3_e2), "(5 / b)")

        div_e1_e3 = e1 / e3
        self.assertIsInstance(div_e1_e3, Divide)
        self.assertEqual(str(div_e1_e3), "(a / 5)")

        rdiv_e2_e1 = e2.__rtruediv__(e1)
        self.assertIsInstance(rdiv_e2_e1, Divide)
        self.assertEqual(str(rdiv_e2_e1), "(a / b)")

    def test_negate_expressions(self):
        e1 = Symbol("x")
        neg_e1 = -e1
        self.assertIsInstance(neg_e1, Negation)
        self.assertEqual(str(neg_e1), "-x")

    def test_and_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        and_e1_e2 = e1 & e2
        self.assertIsInstance(and_e1_e2, And)
        self.assertEqual(str(and_e1_e2), "(a & b)")

        e3 = False
        and_e3_e2 = e3 & e2
        self.assertIsInstance(and_e3_e2, And)
        self.assertEqual(str(and_e3_e2), "(False & b)")

        and_e1_e3 = e1 & e3
        self.assertIsInstance(and_e1_e3, And)
        self.assertEqual(str(and_e1_e3), "(a & False)")

    def test_or_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        or_e1_e2 = e1 | e2
        self.assertIsInstance(or_e1_e2, Or)
        self.assertEqual(str(or_e1_e2), "(a | b)")

        e3 = False
        or_e3_e2 = e3 | e2
        self.assertIsInstance(or_e3_e2, Or)
        self.assertEqual(str(or_e3_e2), "(False | b)")

        or_e1_e3 = e1 | e3
        self.assertIsInstance(or_e1_e3, Or)
        self.assertEqual(str(or_e1_e3), "(a | False)")

    def test_invert_expressions(self):
        e1 = Symbol("a")
        invert_e1 = ~e1
        self.assertIsInstance(invert_e1, Not)
        self.assertEqual(str(invert_e1), "~a")

    def test_greater_than_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        e1_gt_e2 = e1 > e2
        self.assertIsInstance(e1_gt_e2, GreaterThan)
        self.assertEqual(str(e1_gt_e2), "(a > b)")

        e1_gt_10 = e1 > 10
        self.assertIsInstance(e1_gt_10, GreaterThan)
        self.assertEqual(str(e1_gt_10), "(a > 10)")

    def test_greater_than_or_equal_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        e1_ge_e2 = e1 >= e2
        self.assertIsInstance(e1_ge_e2, GreaterThanOrEqual)
        self.assertEqual(str(e1_ge_e2), "(a >= b)")

        e1_ge_10 = e1 >= 10
        self.assertIsInstance(e1_ge_10, GreaterThanOrEqual)
        self.assertEqual(str(e1_ge_10), "(a >= 10)")

    def test_less_than_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        e1_lt_e2 = e1 < e2
        self.assertIsInstance(e1_lt_e2, LessThan)
        self.assertEqual(str(e1_lt_e2), "(a < b)")

        e1_lt_10 = e1 < 10
        self.assertIsInstance(e1_lt_10, LessThan)
        self.assertEqual(str(e1_lt_10), "(a < 10)")

    def test_less_than_or_equal_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        e1_le_e2 = e1 <= e2
        self.assertIsInstance(e1_le_e2, LessThanOrEqual)
        self.assertEqual(str(e1_le_e2), "(a <= b)")

        e1_le_10 = e1 <= 10
        self.assertIsInstance(e1_le_10, LessThanOrEqual)
        self.assertEqual(str(e1_le_10), "(a <= 10)")

    def test_equal_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        e1_eq_e2 = e1 == e2
        self.assertIsInstance(e1_eq_e2, Equal)
        self.assertEqual(str(e1_eq_e2), "(a == b)")

        e1_eq_10 = e1 == 10
        self.assertIsInstance(e1_eq_10, Equal)
        self.assertEqual(str(e1_eq_10), "(a == 10)")

    def test_not_equal_expressions(self):
        e1 = Symbol("a")
        e2 = Symbol("b")
        e1_ne_e2 = e1 != e2
        self.assertIsInstance(e1_ne_e2, NotEqual)
        self.assertEqual(str(e1_ne_e2), "(a != b)")

        e1_ne_10 = e1 != 10
        self.assertIsInstance(e1_ne_10, NotEqual)
        self.assertEqual(str(e1_ne_10), "(a != 10)")

    def test_call_expression(self):
        e1 = Symbol("f")
        call_e1 = e1()
        self.assertIsInstance(call_e1, FunctionCall)
        self.assertEqual(str(call_e1), "f()")

        call_e1_with_args = e1(0, "arg2")
        self.assertIsInstance(call_e1_with_args, FunctionCall)
        self.assertEqual(str(call_e1_with_args), "f(0, arg2)")

        call_e1_with_kwargs = e1(a=0, b="b")
        self.assertIsInstance(call_e1_with_kwargs, FunctionCall)
        self.assertEqual(str(call_e1_with_kwargs), "f(a=0, b=b)")

        call_e1_with_args_kwargs = e1(0, "a2", kw=None)
        self.assertIsInstance(call_e1_with_args_kwargs, FunctionCall)
        self.assertEqual(str(call_e1_with_args_kwargs), "f(0, a2, kw=None)")

        call_e1_with_self = e1(e1)
        self.assertIsInstance(call_e1_with_self, FunctionCall)
        self.assertEqual(str(call_e1_with_self), "f(f)")

        e1.concretize(min)
        call_concrete_f_with_args = e1(0, 1, 2)
        self.assertIsInstance(call_concrete_f_with_args, Constant)
        self.assertEqual(str(call_concrete_f_with_args), "0")

    def test_getattr_expression(self):
        e1 = Symbol("O")
        e2 = Symbol("attr")
        getattr_e1_attr = e1.attr
        self.assertIsInstance(getattr_e1_attr, Attribute)
        self.assertEqual(str(getattr_e1_attr), "O.attr")

        nine4j = Constant(9 + 4j)
        getattr_nine4j_imag = nine4j.imag
        self.assertIsInstance(getattr_nine4j_imag, Constant)
        self.assertEqual(str(getattr_nine4j_imag), "4.0")

        getattr_nine4j_real = getattr(nine4j, "real")
        self.assertIsInstance(getattr_nine4j_real, Constant)
        self.assertEqual(str(getattr_nine4j_real), "9.0")

    def test_getitem_expression(self):
        e1 = Symbol("O")
        index = Symbol("index")
        getitem_e1 = e1[index]
        self.assertIsInstance(getitem_e1, Subscript)
        self.assertEqual(str(getitem_e1), "O[index]")

        getitem_e1_slice = e1[0:3]
        self.assertIsInstance(getitem_e1_slice, Subscript)
        self.assertEqual(str(getitem_e1_slice), "O[0:3]")

        getitem_e1_slice_2 = e1[0:4:2]
        self.assertIsInstance(getitem_e1_slice_2, Subscript)
        self.assertEqual(str(getitem_e1_slice_2), "O[0:4:2]")


class ConstantTests(unittest.TestCase):
    def reset_property_context(self):
        get_context().reset()

    def setUp(self):
        self.reset_property_context()

    def test_singletons(self):
        a1 = Constant("a")
        a2 = Constant("a")
        self.assertIs(a1, a2)
        self.assertEqual(repr(a1), "'a'")
        self.assertEqual(str(a1), "a")

        x1 = Constant(2.2)
        x2 = Constant(2.2)
        self.assertIs(x1, x2)
        x3 = Constant(x1)
        self.assertIs(x1, x3)
        self.assertEqual(repr(x1), "2.2")
        self.assertEqual(str(x1), "2.2")

        list1 = Constant([1, 2, 3])
        list2 = Constant([1, 2, 3])
        self.assertIsNot(list1, list2)

        set1 = Constant({1, 2, 3})
        set2 = Constant({1, 2, 3})
        self.assertIsNot(set1, set2)

        tuple1 = Constant((1, 2, 3))
        tuple2 = Constant((1, 2, 3))
        self.assertIs(tuple1, tuple2)

    def test_bool(self):
        self.assertTrue(Constant(True))
        self.assertFalse(Constant(False))

        self.assertTrue(Constant("a"))
        self.assertFalse(Constant(""))

        self.assertTrue(Constant(12))
        self.assertFalse(Constant(0))

    def test_repr(self):
        dict1 = Constant({"a": 1, "b": 2, "c": 3})
        self.assertEqual(repr(dict1), "{'a': 1, 'b': 2, 'c': 3}")

        list1 = Constant([1, 2, 3])
        self.assertEqual(repr(list1), "[1, 2, 3]")

        set1 = Constant({1, 2, 3})
        self.assertEqual(repr(set1), "{1, 2, 3}")

        tuple1 = Constant((1, 2, 3))
        self.assertEqual(repr(tuple1), "(1, 2, 3)")

        int1 = Constant(2000000)
        self.assertEqual(repr(int1), "2000000")

        float1 = Constant(-32.1004)
        self.assertEqual(repr(float1), "-32.1004")

        slice1 = Constant(slice(0, 3))
        self.assertEqual(repr(slice1), "0:3")
        slice2 = Constant(slice(None, None, 1))
        self.assertEqual(repr(slice2), "::1")

        ndarray = Constant(np.array([-1, 0, 1]))
        # TODO : this is dependent on current implementation, can we make it less so?
        self.assertEqual(repr(ndarray), "<class 'numpy.ndarray'>(id=0x9)")

    def test_str(self):
        dict1 = Constant({"a": 1, "b": 2, "c": 3})
        self.assertEqual(str(dict1), "{'a': 1, 'b': 2, 'c': 3}")

        list1 = Constant([1, 2, 3])
        self.assertEqual(str(list1), "[1, 2, 3]")

        set1 = Constant({1, 2, 3})
        self.assertEqual(str(set1), "{1, 2, 3}")

        tuple1 = Constant((1, 2, 3))
        self.assertEqual(str(tuple1), "(1, 2, 3)")

        int1 = Constant(2000000)
        self.assertEqual(str(int1), "2000000")

        float1 = Constant(-32.1004)
        self.assertEqual(str(float1), "-32.1004")

        slice1 = Constant(slice(0, 3))
        self.assertEqual(str(slice1), "0:3")
        slice2 = Constant(slice(None, None, 1))
        self.assertEqual(str(slice2), "::1")

        ndarray = Constant(np.array([-1, 0, 1]))
        self.assertEqual(str(ndarray), "[-1 0 1]")


if __name__ == "__main__":
    unittest.main()
