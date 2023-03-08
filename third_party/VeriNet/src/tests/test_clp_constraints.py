
"""
Unittests for the CLPConstraint class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.constraints.var import Var


class TestCLPConstraint(unittest.TestCase):

    def setUp(self):

        self._y1 = Var()
        self._y2 = Var()

        self._constraint = 2*self._y1 - self._y2 <= 5
        self._or_constraints = (self._y1 + 2 * self._y2 <= 3) | (self._y1 <= 0)

    def test_as_arrays_single_lp_constraint(self):

        """
        Test that the arrays from of the constraint returned by as_arrays is correct.
        """

        constr_array = self._constraint.as_arrays(self._y1.id, self._y2.id)[0]

        self.assertEqual(2, constr_array[0])
        self.assertEqual(-1, constr_array[1])
        self.assertEqual(-5, constr_array[2])

    def test_as_arrays_conjunction_constraint(self):

        """
        Test that the arrays from of the constraint returned by as_arrays is correct.
        """

        constr_array = self._or_constraints.as_arrays(self._y1.id, self._y2.id)

        self.assertEqual(1, constr_array[0][0])
        self.assertEqual(2, constr_array[0][1])
        self.assertEqual(-3, constr_array[0][2])

        self.assertEqual(1, constr_array[1][0])
        self.assertEqual(0, constr_array[1][1])
        self.assertEqual(0, constr_array[1][2])
