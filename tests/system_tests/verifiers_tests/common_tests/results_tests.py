import unittest

from dnnv.verifiers.common import SAT, UNKNOWN, UNSAT


class VerificationResultsTests(unittest.TestCase):
    def test_and(self):
        self.assertEqual(SAT & SAT, SAT)
        self.assertEqual(SAT & UNSAT, UNSAT)
        self.assertEqual(SAT & UNKNOWN, UNKNOWN)
        self.assertEqual(UNSAT & SAT, UNSAT)
        self.assertEqual(UNSAT & UNSAT, UNSAT)
        self.assertEqual(UNSAT & UNKNOWN, UNSAT)
        self.assertEqual(UNKNOWN & SAT, UNKNOWN)
        self.assertEqual(UNKNOWN & UNSAT, UNSAT)
        self.assertEqual(UNKNOWN & UNKNOWN, UNKNOWN)

        with self.assertRaises(TypeError) as cm:
            _ = UNSAT & 6
        self.assertEqual(
            cm.exception.args[0],
            "unsupported operand type(s) for &: 'PropertyCheckResult' and 'int'",
        )

    def test_or(self):
        self.assertEqual(SAT | SAT, SAT)
        self.assertEqual(SAT | UNSAT, SAT)
        self.assertEqual(SAT | UNKNOWN, SAT)
        self.assertEqual(UNSAT | SAT, SAT)
        self.assertEqual(UNSAT | UNSAT, UNSAT)
        self.assertEqual(UNSAT | UNKNOWN, UNKNOWN)
        self.assertEqual(UNKNOWN | SAT, SAT)
        self.assertEqual(UNKNOWN | UNSAT, UNKNOWN)
        self.assertEqual(UNKNOWN | UNKNOWN, UNKNOWN)

        with self.assertRaises(TypeError) as cm:
            _ = UNSAT | 6
        self.assertEqual(
            cm.exception.args[0],
            "unsupported operand type(s) for |: 'PropertyCheckResult' and 'int'",
        )

    def test_invert(self):
        self.assertEqual(~SAT, UNSAT)
        self.assertEqual(~UNSAT, SAT)
        self.assertEqual(~UNKNOWN, UNKNOWN)

    def test_str(self):
        self.assertEqual(str(SAT), "sat")
        self.assertEqual(str(UNSAT), "unsat")
        self.assertEqual(str(UNKNOWN), "unknown")


if __name__ == "__main__":
    unittest.main()
