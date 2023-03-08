
"""
Unittests for the Var class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.constraints.var import Var


class TestVar(unittest.TestCase):

    def setUp(self):

        self.v1 = Var()
        self.v2 = Var("testname")

    def test_factory(self):

        """
        Tests that the factory method returns the correct amount of variables.
        """

        variables = Var.factory(5)
        self.assertEqual(5, len(variables))
        for i in range(5):
            self.assertTrue(isinstance(variables[i], Var))

    def test_name(self):

        """
        Tests that the correct names are assigned.
        """

        self.assertTrue(self.v2.name == "testname")
        self.v2.name = "V1"
        self.assertTrue(self.v2.name == "V1")

    def test__add__(self):

        """
        Tests the coefficients and constants of the resulting LinEq after addition.
        """

        # Add var and var
        exp1 = self.v1 + self.v2
        exp2 = self.v1 + self.v1
        self.assertEqual(1, exp1.eq_dict[self.v1])
        self.assertEqual(1, exp1.eq_dict[self.v2])
        self.assertEqual(2, exp2.eq_dict[self.v1])

        # Add var and constant
        exp1 = self.v1 + 1
        exp2 = 2 + self.v1
        self.assertEqual(1, exp1.eq_dict[self.v1])
        self.assertEqual(1, exp1.constant)
        self.assertEqual(1, exp2.eq_dict[self.v1])
        self.assertEqual(2, exp2.constant)

    def test__mul__(self):

        """
        Tests the coefficients of the resulting LinEq after multiplication.
        """

        exp1 = 2 * self.v1
        exp2 = self.v1 * 3
        self.assertEqual(2, exp1.eq_dict[self.v1])
        self.assertEqual(3, exp2.eq_dict[self.v1])

    def test__sub__(self):

        """
        Tests the coefficients and constants of the resulting LinEq after subtraction.
        """

        # Subtract var and var
        exp1 = self.v1 - self.v2
        exp2 = self.v1 - self.v1

        self.assertEqual(1, exp1.eq_dict[self.v1])
        self.assertEqual(-1, exp1.eq_dict[self.v2])
        self.assertEqual(0, exp2.eq_dict[self.v1])

        # Subtract var and constant
        exp1 = self.v1 - 1
        exp2 = 2 - self.v1
        self.assertEqual(1, exp1.eq_dict[self.v1])
        self.assertEqual(-1, exp1.constant)
        self.assertEqual(-1, exp2.eq_dict[self.v1])
        self.assertEqual(2, exp2.constant)

    def test__truediv__(self):

        """
        Tests the coefficients and constants of the resulting LinEq after subtraction.
        """

        exp1 = self.v1/5
        self.assertEqual(1/5, exp1.eq_dict[self.v1])
