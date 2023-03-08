
"""
Unit-tests for the ESIP class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import numpy as np
import unittest
import warnings

from src.neural_networks.simple_nn import SimpleNN, SimpleNNConv2, SimpleNNBatchNorm2D
from src.algorithm.esip import ESIP
from src.algorithm.mappings.piecewise_linear import Relu
from src.algorithm.mappings.s_shaped import Sigmoid, Tanh
from src.algorithm.mappings.layers import FC, Conv2d, BatchNorm2d


# noinspection PyArgumentList,PyUnresolvedReferences
class TestNNBounds(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

        self.model_sigmoid = SimpleNN(activation="Sigmoid")
        self.model_tanh = SimpleNN(activation="Tanh")
        self.model_relu = SimpleNN(activation="Relu")
        self.model_conv2 = SimpleNNConv2()
        self.model_batch_norm2d = SimpleNNBatchNorm2D()

        self.bounds_sigmoid = ESIP(self.model_sigmoid, input_shape=2)
        self.bounds_tanh = ESIP(self.model_tanh, input_shape=2)
        self.bounds_relu = ESIP(self.model_relu, input_shape=2)
        self.bounds_conv2 = ESIP(self.model_conv2, input_shape=(1, 5, 5))
        self.bounds_batch_norm_2d = ESIP(self.model_batch_norm2d, input_shape=(1, 2, 2))

    def test_mappings_initialization(self):

        """
        Test that assigned mappings are correct.
        """

        mappings_relu = [None, FC, Relu, FC, Relu]
        for i in range(1, len(mappings_relu)):
            self.assertTrue(isinstance(self.bounds_relu._mappings[i], mappings_relu[i]))

        mappings_sigmoid = [None, FC, Sigmoid, FC, Sigmoid]
        for i in range(1, len(mappings_sigmoid)):
            self.assertTrue(isinstance(self.bounds_sigmoid._mappings[i], mappings_sigmoid[i]))

        mappings_tanh = [None, FC, Tanh, FC, Tanh]
        for i in range(1, len(mappings_tanh)):
            self.assertTrue(isinstance(self.bounds_tanh._mappings[i], mappings_tanh[i]))

        mappings_conv2 = [None, Conv2d, Conv2d]
        for i in range(1, len(mappings_conv2)):
            self.assertTrue(isinstance(self.bounds_conv2._mappings[i], mappings_conv2[i]))

        mappings_batch_norm_2d = [None, Conv2d, BatchNorm2d]
        for i in range(1, len(mappings_batch_norm_2d)):
            self.assertTrue(isinstance(self.bounds_batch_norm_2d._mappings[i], mappings_batch_norm_2d[i]))

    def test_concretize_symbolic_bounds(self):

        """
        Tests the output from the concretize_symbolic_bounds() method against ground truth.
        """

        input_bounds = np.array([[1., 2.], [-2., 0.]])
        symb_bounds = np.array([[1., 1., 0.], [-1., 0., 1.]])
        error_matrix = np.array([[0.5, 1, 1.5], [0, 0, -1]])
        concrete_bounds, errors = self.bounds_relu._calc_bounds_concrete_jit(input_bounds, symb_bounds, error_matrix)

        gt_bounds = np.array([[-1., 5.], [-2, 0.]])
        gt_errors = np.array([[0, 3], [-1, 0]])

        for node_num in range(2):
            self.assertAlmostEqual(gt_bounds[node_num, 0],
                                   concrete_bounds[node_num, 0])
            self.assertAlmostEqual(gt_bounds[node_num, 1],
                                   concrete_bounds[node_num, 1])
            self.assertAlmostEqual(gt_errors[node_num, 0],
                                   errors[node_num, 0])
            self.assertAlmostEqual(gt_errors[node_num, 1],
                                   errors[node_num, 1])

    def test_adjust_bounds_from_forced_bounds(self):

        """
        Tests the output from the _adjust_bounds_from_forced_bounds() method against ground truth.
        """

        concrete_bounds = np.array([[1., 2.], [-2., 0.]])
        forced_bounds = np.array([[-10, 1], [-1, 10]])
        gt_bounds = np.array([[1., 1.], [-1., 0.]])
        concrete_bounds = self.bounds_relu._adjust_bounds_from_forced_bounds(concrete_bounds, forced_bounds)

        for node_num in range(2):
            self.assertAlmostEqual(gt_bounds[node_num, 0],
                                   concrete_bounds[node_num, 0])
            self.assertAlmostEqual(gt_bounds[node_num, 1],
                                   concrete_bounds[node_num, 1])

    def test_valid_concrete_bounds(self):

        """
        Tests the output from the _valid_concrete_bounds() method against ground truth.
        """

        concrete_bounds_valid = np.array([[1, 2], [-1, 2], [3, 5]])
        concrete_bounds_invalid = np.array([[1, -1], [-1, 2], [3, 5]])

        self.assertTrue(self.bounds_relu._valid_concrete_bounds(concrete_bounds_valid))
        self.assertFalse(self.bounds_relu._valid_concrete_bounds(concrete_bounds_invalid))

    def test_calc_relaxations(self):

        """
        Tests the output from the _calc_relaxation() method against ground truth.
        """

        concrete_bounds = np.array([[-2, -1], [-1, 1], [3, 5]])
        gt_relax_lower = np.array([[0, 0], [0.5, 0], [1, 0]])
        gt_relax_upper = np.array([[0, 0], [0.5, 0.5], [1, 0]])
        calc_relax = self.bounds_relu._calc_relaxations(Relu(), concrete_bounds)

        for i in range(gt_relax_upper.shape[0]):
            for j in range(gt_relax_upper.shape[1]):
                self.assertAlmostEqual(gt_relax_lower[i, j], calc_relax[0, i, j])
                self.assertAlmostEqual(gt_relax_upper[i, j], calc_relax[1, i, j])

    def test_prop_equation_trough_relaxation(self):

        """
        Tests the output from the _prop_equation_trough_relaxation() method against ground truth.
        """

        symb_bounds = np.array([[-3, -2, -1]])
        relax = np.array(([[[0.5, 1]], [[0.5, 2]]]))
        gt_new_symb_bounds = np.array([[-1.5, -1, 0.5]])
        new_symb_bounds = self.bounds_relu._prop_equation_trough_relaxation(symb_bounds, relax)

        for i in range(gt_new_symb_bounds.shape[0]):
            for j in range(gt_new_symb_bounds.shape[1]):
                self.assertAlmostEqual(new_symb_bounds[i, j], gt_new_symb_bounds[i, j])

    def test_prop_error_matrix_trough_relaxation(self):

        """
        Tests the output from the _prop_error_matrix_trough_relaxation() method against ground truth.
        """

        error_matrix = np.array([[-3, 1]])
        relax = np.array(([[[0.5, 1]], [[0.5, 2]]]))
        bounds_concrete = np.array([[-1, 1]])
        error_matrix_to_node_indices = np.array([[1, 0], [1, 1]])
        layer_num = 2
        new_error_matrix, error_matrix_to_node_indices = \
            self.bounds_relu._prop_error_matrix_trough_relaxation(error_matrix, relax, bounds_concrete,
                                                                  error_matrix_to_node_indices, layer_num)
        gt_error_matrix = np.array(([[-1.5,  0.5,  1]]))
        gt_error_matrix_to_node_indices = np.array([[1, 0], [1, 1], [2, 0]])

        for i in range(gt_error_matrix.shape[0]):
            for j in range(gt_error_matrix.shape[1]):
                self.assertAlmostEqual(new_error_matrix[i, j], gt_error_matrix[i, j])

        for i in range(gt_error_matrix_to_node_indices.shape[0]):
            for j in range(gt_error_matrix_to_node_indices.shape[1]):
                self.assertAlmostEqual(error_matrix_to_node_indices[i, j], gt_error_matrix_to_node_indices[i, j])

    def test_calculate_symbolic_bounds_sigmoid_brute_force(self):

        """
        Tests that the neural network is within calculated bounds for a range of x values
        """

        x1_range = [-0.5, 1]
        x2_range = [-0.2, 0.6]
        input_constraints = np.array([[x1_range[0], x1_range[1]], [x2_range[0], x2_range[1]]])

        x1_arr = np.linspace(x1_range[0], x1_range[1], 100)
        x2_arr = np.linspace(x2_range[0], x2_range[1], 100)

        self.bounds_sigmoid.calc_bounds(input_constraints)
        bound_symb = self.bounds_sigmoid.bounds_concrete

        # Do a brute force check with different x values
        for x1 in x1_arr:
            for x2 in x2_arr:
                res = self.model_sigmoid.forward(torch.Tensor([[x1, x2]])).detach().numpy()[0, 0]
                self.assertLessEqual(bound_symb[-1][:, 0], res)
                self.assertGreaterEqual(bound_symb[-1][:, 1], res)

    def test_calculate_symbolic_bounds_tanh_brute_force(self):

        """
        Tests that the neural network is within calculated bounds for a range of x values
        """

        x1_range = [-0.5, 1]
        x2_range = [-0.2, 0.6]
        input_constraints = np.array([[x1_range[0], x1_range[1]], [x2_range[0], x2_range[1]]])

        x1_arr = np.linspace(x1_range[0], x1_range[1], 100)
        x2_arr = np.linspace(x2_range[0], x2_range[1], 100)

        self.bounds_tanh.calc_bounds(input_constraints)
        bound_symb = self.bounds_tanh.bounds_concrete

        # Do a brute force check with different x values
        for x1 in x1_arr:
            for x2 in x2_arr:
                res = self.model_tanh.forward(torch.Tensor([[x1, x2]])).detach().numpy()[0, 0]
                self.assertLessEqual(bound_symb[-1][:, 0], res)
                self.assertGreaterEqual(bound_symb[-1][:, 1], res)

    def test_calculate_symbolic_bounds_relu_brute_force(self):

        """
        Tests that the neural network is within calculated bounds for a range of x values
        """

        x1_range = [-1, 1]
        x2_range = [-2, 2]
        input_constraints = np.array([[x1_range[0], x1_range[1]], [x2_range[0], x2_range[1]]])

        x1_arr = np.linspace(x1_range[0], x1_range[1], 100)
        x2_arr = np.linspace(x2_range[0], x2_range[1], 100)

        self.bounds_relu.calc_bounds(input_constraints)
        bound_symb = self.bounds_relu.bounds_concrete

        # Do a brute force check with different x values
        for x1 in x1_arr:
            for x2 in x2_arr:
                res = self.model_relu.forward(torch.Tensor([[x1, x2]])).detach().numpy()[0, 0]
                self.assertLessEqual(bound_symb[-1][:, 0], res)
                self.assertGreaterEqual(bound_symb[-1][:, 1], res)


if __name__ == '__main__':
    unittest.main()
