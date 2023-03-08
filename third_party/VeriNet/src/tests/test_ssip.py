
"""
Unittests for the SSIP class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import unittest
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.sip_torch.ssip import SSIP
from verinet.tests.simple_nn import SimpleNN, SimpleDeepResidual
from verinet.util.config import CONFIG


# noinspection PyArgumentList,PyUnresolvedReferences,DuplicatedCode,PyCallingNonCallable
class TestSSIP(unittest.TestCase):

    # noinspection PyTypeChecker
    def setUp(self):

        self.relu = SimpleNN(activation="ReLU")
        self.deep_residual = SimpleDeepResidual()

        self.bounds_relu = SSIP(self.relu, input_shape=torch.LongTensor((2,)))
        self.bounds_deep_residual = SSIP(self.deep_residual, input_shape=torch.LongTensor((1, 3, 3)))

    def test_prop_bounds_linear(self):

        """
        Tests the _prop_bounds method for a linear node.
        """

        self.bounds_relu._prop_bounds(1)

        gt = torch.FloatTensor([[1, 2, 1], [-1, 2, 0]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(self.bounds_relu.nodes[1].bounds_symbolic_post[0, i, j]), float(gt[i, j]))
                self.assertEqual(float(self.bounds_relu.nodes[1].bounds_symbolic_post[1, i, j]), float(gt[i, j]))

    def test_prop_bounds_non_linear(self):

        """
        Tests the _prop_bounds method for a non-linear node.
        """

        self.bounds_relu.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[[-1, 1], [-2, 2]],
                                                                            [[-1, 1], [-2, 2]]])]

        self.bounds_relu.nodes[1].bounds_symbolic_post = torch.FloatTensor([[[1, 2, 1], [-1, 2, 0]],
                                                                           [[1, 2, 1], [-1, 2, 0]]])
        self.bounds_relu._prop_bounds(2)

        gt = torch.FloatTensor([[[1, 2, 1], [0, 0, 0]], [[0.6, 1.2, 3], [-0.5, 1, 2.5]]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(self.bounds_relu.nodes[2].bounds_symbolic_post[i, j, k]), float(gt[i, j, k]))

    def test_get_pre_act_symb_one_connection(self):

        """
        Tests the _get_pre_act_symb method for one input connection.
        """

        self.bounds_deep_residual.nodes[1].bounds_symbolic_post = torch.FloatTensor([[[1, 2, 1], [-1, 2, 0]],
                                                                                    [[1, 2, 1], [-1, 2, 0]]])

        pre_act_bounds = self.bounds_deep_residual._get_pre_act_symb(2)

        self.assertEqual(len(pre_act_bounds), 1)

        gt = torch.FloatTensor([[[1, 2, 1], [-1, 2, 0]], [[1, 2, 1], [-1, 2, 0]]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(pre_act_bounds[0][i, j, k]), float(gt[i, j, k]))

    def test_get_pre_act_symb_two_connections(self):

        """
        Tests the _get_pre_act_symb method for two input connections.
        """

        self.bounds_deep_residual.nodes[0].bounds_symbolic_post = torch.FloatTensor([[[1, 0, 0], [0, 1, 0]],
                                                                                     [[1, 0, 0], [0, 1, 0]]])
        self.bounds_deep_residual.nodes[3].bounds_symbolic_post = torch.FloatTensor([[[1, 2, 1], [-1, 2, 0]],
                                                                                     [[1, 2, 1], [-1, 2, 0]]])

        pre_act_bounds = self.bounds_deep_residual._get_pre_act_symb(4)

        self.assertEqual(len(pre_act_bounds), 2)

        gt = [torch.FloatTensor([[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]]),
              torch.FloatTensor([[[1, 2, 1], [-1, 2, 0]], [[1, 2, 1], [-1, 2, 0]]])]

        for i in range(gt[0].shape[0]):
            for j in range(gt[0].shape[1]):
                for k in range(gt[0].shape[2]):
                    self.assertEqual(float(pre_act_bounds[0][i, j, k]), float(gt[0][i, j, k]))
                    self.assertEqual(float(pre_act_bounds[1][i, j, k]), float(gt[1][i, j, k]))

    def test_get_pre_concrete_buffered(self):

        """
        Tests the get_pre_concrete method with buffered concrete values.
        """

        gt = torch.FloatTensor([[[1, 2], [0, 1]], [[0, 1], [3, 4]]])
        self.bounds_deep_residual.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[[-1, 1], [-2, 2]],
                                                                                     [[-1, 1], [-2, 2]]])]
        self.bounds_deep_residual.nodes[1].bounds_concrete_pre = [gt]
        bounds_concrete = self.bounds_deep_residual._get_pre_concrete(1)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(bounds_concrete[0][i, j, k]), float(gt[i, j, k]))

    def test_get_pre_concrete_from_symbolic(self):

        """
        Tests the get_pre_concrete method with symbolic values.
        """

        self.bounds_deep_residual.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[[-1, 1], [-2, 2]],
                                                                                     [[-1, 1], [-2, 2]]])]

        self.bounds_deep_residual.nodes[1].bounds_symbolic_pre = [torch.FloatTensor([[[1, 2, 1], [-1, 2, 0]],
                                                                                    [[1, 2, 1], [-1, 2, 0]]])]

        gt = torch.FloatTensor([[[-4, 6], [-5, 5]], [[-4, 6], [-5, 5]]])
        bounds_concrete = self.bounds_deep_residual._get_pre_concrete(1, lazy=False)[0]

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(bounds_concrete[i, j, k]), float(gt[i, j, k]))

    def test_get_pre_concrete_from_symbolic_lazy(self):

        """
        Tests the get_pre_concrete method with symbolic equations using the lazy
        calculations.
        """

        self.bounds_deep_residual.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[[-1, 1], [-2, 2]],
                                                                                     [[-1, 1], [-2, 2]]])]
        self.bounds_deep_residual.nodes[6].forced_bounds_pre = [torch.FloatTensor([[1, 2], [-10, 10]])]

        self.bounds_deep_residual.nodes[6].bounds_symbolic_pre = [torch.FloatTensor([[[1, 2, 1], [-1, 2, 0]],
                                                                                    [[1, 2, 1], [-1, 2, 0]]])]

        gt = torch.FloatTensor([[[1, 2], [-5, 5]], [[1, 2], [-5, 5]]])
        bounds_concrete = self.bounds_deep_residual._get_pre_concrete(6, lazy=True)[0]

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(bounds_concrete[i, j, k]), float(gt[i, j, k]))

    def test_get_post_concrete_from_buffer(self):

        """
        Tests the get_post_concrete method with buffered concrete values.
        """

        gt = torch.FloatTensor([[[1, 2], [0, 1]], [[0, 1], [3, 4]]])
        self.bounds_deep_residual.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[[-1, 1], [-2, 2]],
                                                                                     [[-1, 1], [-2, 2]]])]

        self.bounds_deep_residual.nodes[1].bounds_concrete_post = gt
        bounds_concrete = self.bounds_deep_residual._get_post_concrete(1)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(bounds_concrete[i, j, k]), float(gt[i, j, k]))

    def test_get_post_concrete_from_monothonic_increasing(self):

        """
        Tests the get_post_concrete method with symbolic equations and a monotonically
        increasing operation.
        """

        self.bounds_relu.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[[-1, 1], [-2, 2]],
                                                                            [[-1, 1], [-2, 2]]])]
        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[[1, 2], [-1, 1]], [[0, 1], [3, 4]]])]
        bounds_concrete = self.bounds_relu._get_post_concrete(2)

        gt = torch.FloatTensor([[[1, 2], [0, 1]], [[0, 1], [3, 4]]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(bounds_concrete[i, j, k]), float(gt[i, j, k]))

    def test_get_post_concrete_from_symbolic(self):

        """
        Tests the get_post_concrete method with symbolic equations.
        """

        self.bounds_relu.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[[-1, 1], [-2, 2]],
                                                                            [[-1, 1], [-2, 2]]])]

        self.bounds_relu.nodes[1].bounds_symbolic_post = torch.FloatTensor([[[1, 2, 0], [1, 2, 0]],
                                                                            [[1, 2, 1], [1, 2, -1]]])

        gt = torch.FloatTensor([[[-5, 5], [-5, 5]], [[-4, 6], [-6, 4]]])
        bounds_concrete = self.bounds_relu._get_post_concrete(1)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(bounds_concrete[i, j, k]), float(gt[i, j, k]))

    def test_clean_buffers(self):

        """
        Tests the clean_buffers method.
        """

        CONFIG.STORE_SSIP_BOUNDS = False

        for i in range(4):

            self.bounds_deep_residual.nodes[i].bounds_symbolic_pre = [torch.zeros(1)]
            self.bounds_deep_residual.nodes[i].bounds_symbolic_post = torch.zeros(1)
            self.bounds_deep_residual.nodes[i].relaxations = torch.zeros(1)

        self.bounds_deep_residual._clean_buffers(3)

        for i in range(4):
            self.assertIsNone(self.bounds_deep_residual.nodes[i].bounds_symbolic_pre)
            self.assertIsNone(self.bounds_deep_residual.nodes[i].relaxations)

        for i in [0, 3]:
            self.assertIsNotNone(self.bounds_deep_residual.nodes[i].bounds_symbolic_post)

        for i in [1, 2]:
            self.assertIsNone(self.bounds_deep_residual.nodes[i].bounds_symbolic_post)

    def test_calc_bounds_concrete(self):

        """
        Tests the calc_bounds_concrete method.
        """

        input_bounds = torch.FloatTensor([[-1, 1], [-2, 2]])
        symbolic_bounds = torch.FloatTensor([[1, 2, 0], [1, 2, 1]])

        bounds_concrete = self.bounds_deep_residual._calc_bounds_concrete(input_bounds, symbolic_bounds)

        gt = torch.FloatTensor([[-5, 5], [-4, 6]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(bounds_concrete[i, j]), float(gt[i, j]))

    def test_calc_relaxations(self):

        """
        Tests the calc_relaxation method.
        """

        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[[-2, 1], [-1, 2]],
                                                                            [[-1, 1], [-2, 2]]])]

        relaxations = self.bounds_relu._calc_relaxations(2)

        gt = torch.FloatTensor([[[0, 0], [1, 0]], [[0.5, 0.5], [0.5, 1]]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(relaxations[i, j, k]), float(gt[i, j, k]))

    def test_prop_eq_trough_relaxation(self):

        """
        Tests the _prop_eq_trough_relaxation method.
        """

        bounds_symbolic = torch.FloatTensor([[[1, 2, 0], [1, 2, 0]], [[1, 2, 1], [1, 2, -1]]])
        relaxations = torch.FloatTensor([[[0, 0], [1, 0]], [[0.5, 0.5], [0.4, 0.8]]])

        calculated_bounds_symbolic = self.bounds_relu._prop_eq_trough_relaxation(bounds_symbolic, relaxations)

        gt = torch.FloatTensor([[[0, 0, 0], [1, 2, 0]], [[0.5, 1, 1], [0.4, 0.8, 0.4]]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertAlmostEqual(float(calculated_bounds_symbolic[i, j, k]), float(gt[i, j, k]))

    def test_adjust_bounds_from_forced(self):

        """
        Tests the _prop_eq_trough_relaxation method.
        """

        bounds_concrete = [torch.FloatTensor([[[-2, -1], [-3, -2]], [[1, 2], [2, 3]]])]
        forced = [torch.FloatTensor([[-1.5, 3], [-1, 2.5]])]

        adjusted_bounds = self.bounds_relu._adjust_bounds_from_forced(bounds_concrete, forced)

        gt = torch.FloatTensor([[[-1.5, -1], [-1, -1]], [[1, 2], [2, 2.5]]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertAlmostEqual(float(adjusted_bounds[0][i, j, k]), float(gt[i, j, k]))

    def test_calculate_symbolic_bounds_deep(self):

        """
        Tests that the bounds calculated by SSIP agree with the bounds calculated
        by the model for random inputs.
        """

        input_bounds = torch.zeros((9, 2), dtype=torch.float32)
        input_bounds[:, 1] = 1

        self.bounds_deep_residual.calc_bounds(input_bounds)
        bound_concrete = self.bounds_deep_residual.get_bounds_concrete_post(-1)

        self.assertEqual(bound_concrete[0, 0], 0)
        self.assertEqual(bound_concrete[0, 1], 243)

    def test_calculate_symbolic_bounds_relu_brute_force(self):

        """
        Tests that the bounds calculated by SSIP agree with the bounds calculated
        by the model for random inputs.
        """

        x1_range = [-1, 1]
        x2_range = [-2, 2]
        input_constraints = torch.FloatTensor([[x1_range[0], x1_range[1]], [x2_range[0], x2_range[1]]])

        x1_arr = torch.linspace(x1_range[0], x1_range[1], 10).float()
        x2_arr = torch.linspace(x2_range[0], x2_range[1], 10).float()

        self.bounds_relu.calc_bounds(input_constraints)
        bound_concrete = self.bounds_relu.get_bounds_concrete_post(-1)

        # Do a brute force check with different symb_bounds_in values
        for x1 in x1_arr:
            for x2 in x2_arr:
                res = self.relu.forward(torch.Tensor([[x1, x2]]))[0][0, 0]

                self.assertLessEqual(float(bound_concrete[:, 0]), float(res))
                self.assertGreaterEqual(float(bound_concrete[:, 1]), float(res))
