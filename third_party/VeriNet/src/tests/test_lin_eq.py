
"""
Unittests for the LinEq class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.constraints.var import Var
from verinet.constraints.lin_eq import LinEq


class TestLinEq(unittest.TestCase):

    def setUp(self):

        self.v1 = Var()
        self.v2 = Var("testname")

        self.eq1 = LinEq({self.v1: 1, self.v2: 2})
        self.eq2 = LinEq({self.v1: -1, self.v2: 1}, constant=1)

    def test__add__(self):

        """
        Tests the coefficients and constants of the resulting LinEq after addition.
        """

        # Add constant
        exp1 = self.eq1 + 1
        exp2 = 2 + self.eq1
        self.assertEqual(1, exp1.constant)
        self.assertEqual(2, exp2.constant)

        # Add var
        exp1 = self.eq1 + self.v1
        exp2 = self.v1 + self.eq1
        self.assertEqual(2, exp1.eq_dict[self.v1])
        self.assertEqual(2, exp2.eq_dict[self.v1])

        # Add eq
        exp1 = self.eq1 + self.eq2
        self.assertEqual(0, exp1.eq_dict[self.v1])
        self.assertEqual(3, exp1.eq_dict[self.v2])
        self.assertEqual(1, exp1.constant)

    def test__mul__(self):

        """
        Tests the coefficients and constants of the resulting LinEq after
        multiplication.
        """

        exp1 = self.eq2 * 3
        self.assertEqual(-3, exp1.eq_dict[self.v1])
        self.assertEqual(3, exp1.eq_dict[self.v2])
        self.assertEqual(3, exp1.constant)

        exp2 = 2 * self.eq1
        self.assertEqual(2, exp2.eq_dict[self.v1])
        self.assertEqual(4, exp2.eq_dict[self.v2])
        self.assertEqual(0, exp2.constant)

    def test__sub__(self):

        """
        Tests the coefficients and constants of the resulting LinEq after subtraction.
        """

        # Subtract constant
        exp1 = self.eq1 - 1
        self.assertEqual(1, exp1.eq_dict[self.v1])
        self.assertEqual(2, exp1.eq_dict[self.v2])
        self.assertEqual(-1, exp1.constant)

        exp2 = 2 - self.eq2
        self.assertEqual(1, exp2.eq_dict[self.v1])
        self.assertEqual(-1, exp2.eq_dict[self.v2])
        self.assertEqual(1, exp2.constant)

        # Subtract var
        exp1 = self.eq1 - self.v1
        self.assertEqual(0, exp1.eq_dict[self.v1])
        self.assertEqual(2, exp1.eq_dict[self.v2])
        self.assertEqual(0, exp1.constant)

        exp2 = self.v1 - self.eq2
        self.assertEqual(2, exp2.eq_dict[self.v1])
        self.assertEqual(-1, exp2.eq_dict[self.v2])
        self.assertEqual(-1, exp2.constant)

        # Add eq
        exp1 = self.eq1 - self.eq2
        self.assertEqual(2, exp1.eq_dict[self.v1])
        self.assertEqual(1, exp1.eq_dict[self.v2])
        self.assertEqual(-1, exp1.constant)

    def test__truediv__(self):

        """
        Tests the coefficients and constants of the resulting LinEq after
        multiplication.
        """

        exp1 = self.eq2/5
        self.assertEqual(-1/5, exp1.eq_dict[self.v1])
        self.assertEqual(1/5, exp1.eq_dict[self.v2])
        self.assertEqual(1/5, exp1.constant)

    def test__lq__(self):

        """
        Tests that the correct constraint is returned of the lqs operation.
        """

        constr1 = self.eq1 <= self.eq2
        self.assertEqual(2, constr1._lin_eqs[0].eq_dict[self.v1])
        self.assertEqual(1, constr1._lin_eqs[0].eq_dict[self.v2])
        self.assertEqual(-1, constr1._lin_eqs[0].constant)

        constr2 = 1 <= self.eq2
        self.assertEqual(1, constr2._lin_eqs[0].eq_dict[self.v1])
        self.assertEqual(-1, constr2._lin_eqs[0].eq_dict[self.v2])
        self.assertEqual(0, constr2._lin_eqs[0].constant)
