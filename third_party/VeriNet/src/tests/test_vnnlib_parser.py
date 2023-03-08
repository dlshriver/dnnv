
"""
Unittests for the CLPConstraint class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

import numpy as np

from verinet.parsers.vnnlib_parser import VNNLIBParser
from verinet.tests.simple_nn import SimpleNN


class TestVnnlibParser(unittest.TestCase):

    def test_global_in_global_out(self):

        parser = VNNLIBParser("./vnnlib_samples/prop_1.vnnlib")
        model = SimpleNN(activation="ReLU")

        objectives = parser.get_objectives_from_vnnlib(model, input_shape=(3,))

        self.assertEqual(len(objectives), 1)
        self.assertEqual(len(objectives[0]._constraints), 1)

        objective = objectives[0]

        input_bounds = objective.input_bounds
        output_bounds = objective._constraints[0].as_arrays(objective._first_var_id, objective._last_var_id)
        gt_input = np.array([[0.5, 1], [0, 0.5], [-0.5, 0]])
        gt_output = np.array([[1, 0, 0, -1], [0, -1, 0, 2], [0, -1, 1, 0]])

        for i in range(gt_input.shape[0]):
            for j in range(gt_input.shape[1]):
                self.assertEqual(input_bounds[i, j], gt_input[i, j])

        for i in range(gt_output.shape[0]):
            for j in range(gt_output.shape[1]):
                self.assertEqual(output_bounds[i][j], gt_output[i, j])

    def test_global_in_or_out(self):

        parser = VNNLIBParser("./vnnlib_samples/prop_2.vnnlib")
        model = SimpleNN(activation="ReLU")

        objectives = parser.get_objectives_from_vnnlib(model, input_shape=(3,))

        self.assertEqual(len(objectives), 1)
        self.assertEqual(len(objectives[0]._constraints), 3)

        objective = objectives[0]

        input_bounds = objective.input_bounds

        output_bounds = [constr.as_arrays(objective._first_var_id, objective._last_var_id) for constr
                         in objective._constraints]

        gt_input = np.array([[0.5, 1], [0, 0.5], [-0.5, 0]])
        gt_output = np.array([[-1, 0, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]])

        for i in range(gt_input.shape[0]):
            for j in range(gt_input.shape[1]):
                self.assertEqual(input_bounds[i, j], gt_input[i, j])

        for i in range(3):
            self.assertEqual(len(output_bounds[i]), 1)
            for j in range(4):
                self.assertEqual(output_bounds[i][0][j], gt_output[i, j])

    def test_or_in_global_out(self):

        parser = VNNLIBParser("./vnnlib_samples/prop_3.vnnlib")
        model = SimpleNN(activation="ReLU")

        objectives = parser.get_objectives_from_vnnlib(model, input_shape=(3,))

        self.assertEqual(len(objectives), 2)
        self.assertEqual(len(objectives[0]._constraints), 1)
        self.assertEqual(len(objectives[1]._constraints), 1)

        gt_inputs = [np.array([[0.5, 1], [0, 0.5], [-0.5, 0]]), np.array([[1, 1.5], [0.5, 1], [0, 0.5]])]
        gt_output = np.array([[1, 0, 0, -1], [0, -1, 0, 2], [0, -1, 1, 0]])

        for k, objective in enumerate(objectives):

            input_bounds = objective.input_bounds
            output_bounds = objective._constraints[0].as_arrays(objective._first_var_id, objective._last_var_id)
            gt_input = gt_inputs[k]

            for i in range(gt_input.shape[0]):
                for j in range(gt_input.shape[1]):
                    self.assertEqual(input_bounds[i, j], gt_input[i, j])

            for i in range(gt_output.shape[0]):
                for j in range(gt_output.shape[1]):
                    self.assertEqual(output_bounds[i][j], gt_output[i, j])

    def test_or_in_or_out(self):

        parser = VNNLIBParser("./vnnlib_samples/prop_4.vnnlib")
        model = SimpleNN(activation="ReLU")

        objectives = parser.get_objectives_from_vnnlib(model, input_shape=(3,))

        self.assertEqual(len(objectives), 2)
        self.assertEqual(len(objectives[0]._constraints), 3)
        self.assertEqual(len(objectives[1]._constraints), 3)

        gt_inputs = [np.array([[0.5, 1], [0, 0.5], [-0.5, 0]]), np.array([[1, 1.5], [0.5, 1], [0, 0.5]])]
        gt_output = np.array([[-1, 0, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]])

        for k, objective in enumerate(objectives):

            input_bounds = objective.input_bounds
            output_bounds = [constr.as_arrays(objective._first_var_id, objective._last_var_id) for constr
                             in objective._constraints]
            gt_input = gt_inputs[k]

            for i in range(gt_input.shape[0]):
                for j in range(gt_input.shape[1]):
                    self.assertEqual(input_bounds[i, j], gt_input[i, j])

            for i in range(3):
                self.assertEqual(len(output_bounds[i][0]), 4)
                for j in range(4):
                    self.assertEqual(output_bounds[i][0][j], gt_output[i, j])

    def test_long_and_in_or_out(self):

        parser = VNNLIBParser("./vnnlib_samples/prop_5.vnnlib")
        model = SimpleNN(activation="ReLU")

        objectives = parser.get_objectives_from_vnnlib(model, input_shape=(3,))
        self.assertEqual(len(objectives), 1)
        self.assertEqual(len(objectives[0]._constraints), 3)

        objective = objectives[0]

        input_bounds = objective.input_bounds

        output_bounds = [constr.as_arrays(objective._first_var_id, objective._last_var_id) for constr
                         in objective._constraints]

        gt_input = np.array([[0.5, 1], [0, 0.5], [-0.5, 0]])
        gt_output = [[[-1, 1, 0, 0], [-1, 0, 1, 0], [1, 0, 0, -2]], [[0, -1, 1, 0]], [[0, 1, 0, -1]]]

        for i in range(gt_input.shape[0]):
            for j in range(gt_input.shape[1]):
                self.assertEqual(input_bounds[i, j], gt_input[i, j])
        print("OK")
        for i in range(len(gt_output)):
            for j in range(len(gt_output[i])):
                for k in range(len(gt_output[i][j])):
                    self.assertEqual(output_bounds[i][j][k], gt_output[i][j][k])

    def test_mixed(self):

        parser = VNNLIBParser("./vnnlib_samples/prop_6.vnnlib")
        model = SimpleNN(activation="ReLU")

        objectives = parser.get_objectives_from_vnnlib(model, input_shape=(3,))

        self.assertEqual(len(objectives), 2)
        self.assertEqual(len(objectives[0]._constraints), 1)
        self.assertEqual(len(objectives[1]._constraints), 1)

        gt_inputs = [np.array([[0.5, 1], [0, 0.5], [-0.5, 0]]), np.array([[1, 1.5], [0.5, 1], [0, 0.5]])]
        gt_output = np.array([[-1, 1, 0, 0], [0, -1, 1, 0]])

        for k, objective in enumerate(objectives):

            input_bounds = objective.input_bounds
            output_bounds = [constr.as_arrays(objective._first_var_id, objective._last_var_id) for constr
                             in objective._constraints]
            gt_input = gt_inputs[k]

            for i in range(gt_input.shape[0]):
                for j in range(gt_input.shape[1]):
                    self.assertEqual(input_bounds[i, j], gt_input[i, j])

            self.assertEqual(len(output_bounds[0][0]), 4)
            for j in range(4):
                self.assertEqual(output_bounds[0][0][j], gt_output[k, j])
