
"""
Unittests for the RSIP class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import unittest
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.sip_torch.rsip import RSIP
from verinet.tests.simple_nn import SimpleNN, SimpleDeepResidual, SimpleNN2Outputs, SimpleNN2Layers


# noinspection PyArgumentList,PyUnresolvedReferences,DuplicatedCode,PyCallingNonCallable
class TestRSIP(unittest.TestCase):

    # noinspection PyTypeChecker
    def setUp(self):
        CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS = False

        self.relu = SimpleNN(activation="ReLU")
        self.relu_2_outputs = SimpleNN2Outputs(activation="ReLU")
        self.relu_2_layers = SimpleNN2Layers(activation="ReLU")
        self.deep_residual = SimpleDeepResidual()

        self.bounds_relu = RSIP(self.relu, input_shape=torch.LongTensor((2,)))
        self.bounds_relu_2_outputs = RSIP(self.relu_2_outputs, input_shape=torch.LongTensor((2,)))
        self.bounds_relu_2_layers = RSIP(self.relu_2_layers, input_shape=torch.LongTensor((2,)))
        self.bounds_deep_residual = RSIP(self.deep_residual, input_shape=torch.LongTensor((1, 3, 3)))

    def test_calc_concrete_bounds_pre(self):

        """
        Tests the _calc_concrete_bounds_pre method for a node.
        """
        self.bounds_relu._store_intermediate_bounds = False

        self.bounds_relu.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[-1, 1], [-1, 1]])]
        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[-10, 10], [-10, 10]])]

        self.bounds_relu._calc_concrete_bounds_pre(2, optimise_computations=False)

        gt = torch.FloatTensor([[-2, 4], [-3, 3]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(self.bounds_relu.nodes[2].bounds_concrete_pre[0][i, j]), float(gt[i, j]))
                self.assertEqual(float(self.bounds_relu.nodes[2].bounds_concrete_pre[0][i, j]), float(gt[i, j]))

    def test_calc_concrete_bounds_pre_optimised(self):

        """
        Tests the _calc_concrete_bounds_pre method for a node with optimised computations.
        """

        self.bounds_relu.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[-1, 1], [-1, 1]])]
        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[1, 10], [-10, 10]])]
        self.bounds_relu._calc_concrete_bounds_pre(2)

        gt = torch.FloatTensor([[1, 10], [-3, 3]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(self.bounds_relu.nodes[2].bounds_concrete_pre[0][i, j]), float(gt[i, j]))
                self.assertEqual(float(self.bounds_relu.nodes[2].bounds_concrete_pre[0][i, j]), float(gt[i, j]))

    def test_calc_concrete_bounds_post(self):

        """
        Tests the _calc_concrete_bounds_pre method for a node with optimised computations.
        """

        input_constraints = torch.FloatTensor([[-1, 1], [-2, 2]])
        self.bounds_relu.nodes[0].bounds_concrete_pre = [input_constraints]
        self.bounds_relu.nodes[2].bounds_concrete_pre = [input_constraints]
        self.bounds_relu.nodes[2].relaxations = torch.FloatTensor([[[1, 0], [0, 0]], [[0.6, 2.4], [0.5, 2.5]]])

        bounds_concrete = self.bounds_relu.get_bounds_concrete_post(-1, force_recalculate=True)

        self.assertAlmostEqual(float(bounds_concrete[0, 0]), -7)
        self.assertAlmostEqual(float(bounds_concrete[0, 1]), 4)

    def test_get_non_linear_neurons(self):

        """
        Tests the _get_non_linear_neurons method for a node.
        """

        self.bounds_relu.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[-1, 1], [-1, 1]])]
        self.bounds_relu.nodes[1].bounds_concrete_post = torch.FloatTensor([[1, 10], [-10, 10], [-1, -2]])
        non_linear_nodes, _ = self.bounds_relu._get_non_linear_neurons(2, optimise_computations=False)
        self.assertIsNone(non_linear_nodes)

    def test_get_non_linear_neurons_optimised(self):

        """
        Tests the _get_non_linear_neurons method for a node with optimised computations.
        """

        self.bounds_relu.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[-1, 1], [-1, 1]])]
        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[1, 10], [-10, 10], [-1, -2]])]
        non_linear_nodes, _ = self.bounds_relu._get_non_linear_neurons(2)

        self.assertEqual(len(non_linear_nodes), 1)
        self.assertEqual(int(non_linear_nodes[0]), 1)

    def test_calc_weak_post_concrete_bounds_linear(self):

        """
        Tests the _calc_weak_post_concrete_bounds method for a linear node.
        """

        self.bounds_relu.nodes[1].bounds_concrete_pre = [torch.FloatTensor([[-1, 2], [-1, 2]])]
        self.bounds_relu._calc_weak_post_concrete_bounds(1)

        gt = torch.FloatTensor([[-2, 7], [-4, 5]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(self.bounds_relu.nodes[1].bounds_concrete_post[i, j]), float(gt[i, j]))
                self.assertEqual(float(self.bounds_relu.nodes[1].bounds_concrete_post[i, j]), float(gt[i, j]))

    def test_calc_weak_post_concrete_bounds_non_linear(self):

        """
        Tests the _calc_weak_post_concrete_bounds method for a linear node.
        """

        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[-1, 2], [-1, 2]])]
        self.bounds_relu.nodes[2].relaxations = torch.FloatTensor([[[0, 0], [1, 0]], [[0.5, 0.5], [0.75, 0.5]]])

        self.bounds_relu._calc_weak_post_concrete_bounds(2)

        gt = torch.FloatTensor([[0, 1.5], [-1, 2]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertAlmostEqual(float(self.bounds_relu.nodes[2].bounds_concrete_post[i, j]), float(gt[i, j]))
                self.assertAlmostEqual(float(self.bounds_relu.nodes[2].bounds_concrete_post[i, j]), float(gt[i, j]))

    def test_calc_weak_pre_concrete_bounds(self):

        """
        Tests the _calc_weak_post_concrete_bounds method for a node.
        """

        self.bounds_deep_residual.nodes[0].bounds_concrete_post = torch.FloatTensor([[-1, 2], [-1, 2]])
        self.bounds_deep_residual.nodes[3].bounds_concrete_post = torch.FloatTensor([[0, 1], [0, 1]])

        res = self.bounds_deep_residual._calc_weak_pre_concrete_bounds(4)
        gt = [torch.FloatTensor([[-1, 2], [-1, 2]]), torch.FloatTensor([[0, 1], [0, 1]])]

        for k in range(len(gt)):
            for i in range(gt[k].shape[0]):
                for j in range(gt[k].shape[1]):
                    self.assertAlmostEqual(float(res[k][i, j]), float(gt[k][i, j]))
                    self.assertAlmostEqual(float(res[k][i, j]), float(gt[k][i, j]))

    def test_get_mem_limited_node_indices(self):

        """
        Tests the _get_mem_limited_node_indices method.
        """

        self.bounds_relu._max_estimated_memory_usage = 16
        indices = self.bounds_relu._get_mem_limited_node_indices(2, 5)

        gt = [0, 2, 4, 5]

        for i in range(len(gt)):
            self.assertEqual(indices[i], gt[i])

    def test_backprop_symb_equations_post_linear(self):

        """
        Tests the _backprop_symb_equations method.
        """

        bounds_symbolic_post = torch.FloatTensor([[1, 2, 1],
                                                  [-1, -2, 0]])

        res = self.bounds_relu._backprop_symb_equations(1, bounds_symbolic_post=bounds_symbolic_post)
        gt = torch.FloatTensor([[-1, 6, 2], [1, -6, -1]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(res[0][i, j]), float(gt[i, j]))

    def test_backprop_symb_equations_post_non_linear(self):

        """
        Tests the _backprop_symb_equations method.
        """

        bounds_symbolic_post = torch.FloatTensor([[2, 4, 0.5],
                                                  [-1, -2, 0]])
        self.bounds_relu.nodes[2].relaxations = torch.FloatTensor([[[1, 0], [1, 0]], [[0.5, 0.25], [0.5, 0]]])
        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[-10, 10], [-10, 10]])]

        res = self.bounds_relu._backprop_symb_equations(2, bounds_symbolic_post=bounds_symbolic_post, lower=False)
        gt = torch.FloatTensor([[-1, 6, 2], [1, -6, -1]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(res[0][i, j]), float(gt[i, j]))

    def test_backprop_symb_equations_pre_linear(self):

        """
        Tests the _backprop_symb_equations method.
        """

        bounds_symbolic_pre = [torch.FloatTensor([[1, 2, 1],
                                                  [-1, -2, 0]])]

        res = self.bounds_relu._backprop_symb_equations(2, bounds_symbolic_pre=bounds_symbolic_pre)
        gt = torch.FloatTensor([[-1, 6, 2], [1, -6, -1]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(res[0][i, j]), float(gt[i, j]))

    def test_adjust_bounds_from_forced(self):

        """
        Tests the _adjust_bounds_from_forced method.
        """

        bounds_concrete = [torch.FloatTensor([[-1, 1], [-2, 1], [-1, 2], [-2, 2]])]
        forced = [torch.FloatTensor([[-10, 10], [-1, 10], [-10, 1], [-1, 1]])]

        adjusted_bounds = self.bounds_relu._adjust_bounds_from_forced(bounds_concrete, forced)

        gt = torch.FloatTensor([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertAlmostEqual(float(adjusted_bounds[0][i, j]), float(gt[i, j]))

    def test_get_node_bounding_equation_linear(self):

        """
        Tests the get_node_bounding_equation method for only linear nodes.
        """

        res = self.bounds_relu.get_neuron_bounding_equation(2, 0, lower=True)

        gt = torch.FloatTensor([1, 2, 1])

        for i in range(len(gt)):
            self.assertAlmostEqual(float(res[0][0, i]), float(gt[i]))

    def test_get_node_bounding_equation_non_linear(self):

        """
        Tests the get_node_bounding_equation for a non-linear back-prop.
        """

        self.bounds_relu.nodes[2].relaxations = torch.FloatTensor([[[1, 0], [1, 0]], [[0.5, 0.25], [0.5, 0]]])
        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[-10, 10], [-10, 10]])]
        res = self.bounds_relu.get_neuron_bounding_equation(3, 0, lower=False)

        gt = torch.FloatTensor([0.5, 1, 0.75])

        for i in range(len(gt)):
            self.assertAlmostEqual(float(res[0][0, i]), float(gt[i]))

    def test_get_node_bounding_equation_separate_bias(self):

        """
        Tests the get_node_bounding_equation with separate_bias.
        """
        CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS = True

        self.bounds_relu.nodes[2].relaxations = torch.FloatTensor([[[1, 0], [1, 0]],
                                                                   [[0.5, 0.25], [0.5, 0.5]]])

        self.bounds_relu.nodes[2].relaxations_parallel = torch.FloatTensor([[[0.5, 0], [0.5, 0]],
                                                                            [[0.5, 0.25], [0.5, 0.5]]])

        self.bounds_relu.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[-10, 10], [-10, 10]])]
        res = self.bounds_relu.get_neuron_bounding_equation(3, 0, lower=False, bias_sep_constraints=True)

        gt = torch.FloatTensor([0.5, 1, 0.25, 0.0, 0.5])
        for i in range(len(gt)):
            self.assertAlmostEqual(float(res[0][0, i]), float(gt[i]))

    def test_convert_output_bounding_equation(self):

        """
        Tests the convert_output_bounding_equation method.
        """

        output_equations = torch.FloatTensor([[2, 4], [-1, -2]])
        self.bounds_relu_2_outputs.nodes[2].relaxations = torch.FloatTensor([[[1, 0], [1, 0]], [[0.5, 0.25], [0.5, 0]]])
        self.bounds_relu_2_outputs.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[-10, 10], [-10, 10]])]

        res = self.bounds_relu_2_outputs.convert_output_bounding_equation(output_equations=output_equations,
                                                                          lower=False)
        gt = torch.FloatTensor([[-1, 6, 1.5], [1, -6, -1]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(res[i, j]), float(gt[i, j]))

    def test_convert_output_bounding_equation_separate_biases(self):

        """
        Tests the convert_output_bounding_equation method with separated biases.
        """

        output_equations = torch.FloatTensor([[2, 4], [-1, -2]])
        self.bounds_relu_2_outputs.nodes[2].relaxations = torch.FloatTensor([[[1, 0], [1, 0]],
                                                                             [[0.5, 0.25], [0.5, 0]]])

        self.bounds_relu_2_outputs.nodes[2].relaxations_parallel = torch.FloatTensor([[[0.5, 0], [0.5, 0]],
                                                                                      [[0.5, 0.25], [0.5, 1]]])

        self.bounds_relu_2_outputs.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[-10, 10], [-10, 10]])]

        res = self.bounds_relu_2_outputs.convert_output_bounding_equation(output_equations=output_equations,
                                                                          lower=False,
                                                                          bias_sep_constraints=True)
        gt = torch.FloatTensor([[-1, 6, 0.5, 4, 1], [0.5, -3, -0.25, -2, -0.5]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(res[i, j]), float(gt[i, j]))

    def test_convert_output_bounding_equation_separate_biases_1_node(self):

        """
        Tests the convert_output_bounding_equation method with separated biases.
        """

        CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS = True

        output_equations = torch.FloatTensor([[2, 4], [-1, -2]])
        self.bounds_relu_2_outputs.nodes[2].relaxations = torch.FloatTensor([[[1, 0], [0, 0]],
                                                                             [[1, 0], [1, 1]]])

        self.bounds_relu_2_outputs.nodes[2].relaxations_parallel = torch.FloatTensor([[[1, 0], [0.5, 0]],
                                                                                      [[1, 0], [0.5, 1]]])

        self.bounds_relu_2_outputs.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[1, 10], [-10, 10]])]

        res = self.bounds_relu_2_outputs.convert_output_bounding_equation(output_equations=output_equations,
                                                                          lower=False,
                                                                          bias_sep_constraints=True)
        gt = torch.FloatTensor([[0, 8, 4, 2], [0, -4, -2, -1]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(res[i, j]), float(gt[i, j]))

    def test_separate_bias(self):

        """
        Tests the _separate_bias method.
        """

        self.bounds_relu_2_outputs.nodes[3].biases = {2: torch.FloatTensor([[1, 2], [3, 0]])}
        self.bounds_relu_2_outputs.nodes[3].relax_diff = {2: torch.FloatTensor([[1, 2], [3, 4]])}
        self.bounds_relu_2_outputs.nodes[3].biases_sum = {2: torch.FloatTensor([3, 3])}

        output_equations = torch.FloatTensor([[2, 4, 6], [-1, -2, -3]])
        res = self.bounds_relu_2_outputs.separate_bias(3, output_equations)
        gt = torch.FloatTensor([[2., 4., 1., 2., 3.],
                                [-1., -2., 3., 4., -6.]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(res[i, j], gt[i, j])

    def test_set_parallel_relaxations(self):

        """
        Tests the set_parallel_relaxations method.
        """

        self.bounds_relu_2_layers.nodes[2].relaxations = torch.FloatTensor([[[1, 0], [1, 0]],
                                                                            [[0.5, 0.25], [0.5, 0]]])

        self.bounds_relu_2_layers.nodes[2].relaxations_parallel = torch.FloatTensor([[[0.5, 0], [0.5, 0]],
                                                                                     [[0.5, 0.5], [0.5, 0]]])
        self.bounds_relu_2_layers.set_parallel_relaxations()

        gt = self.bounds_relu_2_layers.nodes[2].relaxations_parallel

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(self.bounds_relu_2_layers.nodes[2].relaxations[i, j, k]), float(gt[i, j, k]))

    def test_set_non_parallel_relaxations(self):

        """
        Tests the set_non_parallel_relaxations method.
        """

        self.bounds_relu_2_layers.nodes[2].relaxations = torch.FloatTensor([[[0.5, 0], [0.5, 0]],
                                                                            [[0.5, 0.5], [0.5, 0]]])

        self.bounds_relu_2_layers.nodes[2].relaxations_non_parallel = torch.FloatTensor([[[1, 0], [1, 0]],
                                                                                         [[0.5, 0.25], [0.5, 0]]])

        self.bounds_relu_2_layers.set_non_parallel_relaxations()

        gt = self.bounds_relu_2_layers.nodes[2].relaxations_non_parallel

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertEqual(float(self.bounds_relu_2_layers.nodes[2].relaxations[i, j, k]), float(gt[i, j, k]))

    def test_calc_input_node_simple_impact(self):

        """
        Tests the _calc_input_node_simple_impact method.
        """

        output_equation = torch.FloatTensor([[2, 4]])

        self.bounds_relu_2_outputs.nodes[0].bounds_concrete_pre = [torch.FloatTensor([[-0.5, 0.5], [-1, 1]])]
        self.bounds_relu_2_outputs.nodes[2].bounds_concrete_pre = [torch.FloatTensor([[-1, 1], [-2, 2]])]
        self.bounds_relu_2_outputs.nodes[2].relaxations = torch.FloatTensor([[[0, 0], [0, 0]],
                                                                            [[1 / 2, 1 / 2], [1 / 2, 1 / 2]]])

        gt_scores = torch.FloatTensor([0.5, 6])

        self.bounds_relu_2_outputs._calc_input_node_simple_impact(output_equation, lower=False)
        scores = self.bounds_relu_2_outputs.nodes[0].impact

        for i in range(2):
            self.assertAlmostEqual(float(gt_scores[i]), float(scores[i]))

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
        bounds_concrete = self.bounds_relu.get_bounds_concrete_post(-1, force_recalculate=True)

        # Do a brute force check with different symb_bounds_in values
        for x1 in x1_arr:
            for x2 in x2_arr:
                res = self.relu.forward(torch.Tensor([[x1, x2]]))[0][0, 0]
                self.assertLessEqual(float(bounds_concrete[:, 0]), float(res))
                self.assertGreaterEqual(float(bounds_concrete[:, 1]), float(res))
