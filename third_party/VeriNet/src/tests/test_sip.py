"""
Unittests for the SIP class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import unittest
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.sip_torch.sip import SIP
from verinet.tests.simple_nn import SimpleDeepResidual
from verinet.sip_torch.operations.linear import Identity, Conv2d, MulConstant, AddDynamic
from verinet.sip_torch.operations.piecewise_linear import Relu


# noinspection PyCallingNonCallable
class TestSIP(unittest.TestCase):

    def setUp(self):

        self.deepResidual = SimpleDeepResidual()
        self.sip = SIP(self.deepResidual, input_shape=torch.LongTensor((1, 3, 3)))

    # noinspection PyTypeChecker
    def test_read_nodes_from_torch_model(self):

        """
        Test reading the nodes from a deep residual torch model.
        """

        shape_hidden = (1, 3, 3)
        shape_output = (1, 1, 1)

        ops = [Identity, Conv2d, Relu, MulConstant, AddDynamic, Conv2d, Relu, MulConstant, AddDynamic,
               Conv2d, Relu, MulConstant, AddDynamic, Conv2d]

        for i, node in enumerate(self.sip.nodes):

            self.assertTupleEqual(tuple(node.in_shape), shape_hidden)
            if i < 13:
                self.assertTupleEqual(tuple(node.out_shape), shape_hidden)
            else:
                self.assertTupleEqual(tuple(node.out_shape), shape_output)

            self.assertIsInstance(node.op, ops[i])

            self.assertTupleEqual(tuple(node.connections_to), tuple(self.deepResidual.nodes[i].connections_to))
            self.assertTupleEqual(tuple(node.connections_from), tuple(self.deepResidual.nodes[i].connections_from))

    # noinspection PyTypeChecker
    def test_get_split_point(self):

        """
        Test that the get_split_point method returns the correct values for input
        and ReLU nodes.
        """

        self.assertEqual(self.sip.get_split_point(-2, 4, node_num=0), 1)
        self.assertEqual(self.sip.get_split_point(-2, 4, node_num=2), 0)

    # noinspection PyTypeChecker
    def test_merge_bounds_into_forced(self):

        """
        Test that the concrete bounds are merged correctly into forced bounds.
        """

        forced = [[torch.zeros((9, 2))], [torch.zeros((9, 2))]] + [None] * 12
        forced[0][0][:, 1] = 1
        forced[1][0][:, 0] = 0.5
        forced[1][0][:, 1] = 0.6

        concrete = [[torch.cat((torch.zeros(9, 1), torch.ones(9, 1)), dim=1)] for _ in range(14)]
        concrete[0][0][0, 0] = 0.5
        concrete[0][0][0, 1] = 0.6
        concrete[0][0][1, 0] = 0.4
        concrete[0][0][2, 1] = 0.7

        for i, node in enumerate(self.sip.nodes):
            node.bounds_concrete_pre = concrete[i]
        self.sip.set_forced_bounds_pre(forced)

        self.sip.merge_current_bounds_into_forced()

        self.assertEqual(self.sip.get_forced_bounds_pre()[0][0][0, 0], concrete[0][0][0, 0])
        self.assertEqual(self.sip.get_forced_bounds_pre()[0][0][0, 1], concrete[0][0][0, 1])
        self.assertEqual(self.sip.get_forced_bounds_pre()[0][0][1, 0], concrete[0][0][1, 0])
        self.assertEqual(self.sip.get_forced_bounds_pre()[0][0][1, 1], forced[0][0][1, 1])
        self.assertEqual(self.sip.get_forced_bounds_pre()[0][0][2, 1], concrete[0][0][2, 1])
        self.assertEqual(self.sip.get_forced_bounds_pre()[0][0][2, 0], forced[0][0][2, 0])

        for i in range(3, 9):
            self.assertEqual(self.sip.get_forced_bounds_pre()[0][0][i, 0], 0)
            self.assertEqual(self.sip.get_forced_bounds_pre()[0][0][i, 1], 1)

        for i in range(9):
            self.assertEqual(self.sip.get_forced_bounds_pre()[1][0][i, 0], 0.5)
            self.assertEqual(self.sip.get_forced_bounds_pre()[1][0][i, 1], 0.6)

        for layer_num in range(2, 13):
            for i in range(9):
                self.assertEqual(self.sip.get_forced_bounds_pre()[layer_num][0][i, 0], 0)
                self.assertEqual(self.sip.get_forced_bounds_pre()[layer_num][0][i, 1], 1)

    # noinspection PyTypeChecker
    def test_valid_concrete_bounds(self):

        """
        Test that the concrete bounds are merged correctly into forced bounds.
        """

        bounds_valid = torch.FloatTensor([[0, 1],
                                          [0, 1]])

        bounds_round = torch.FloatTensor([[0.5+1e-7, 0.5-1e-7],
                                          [0, 1]])

        bounds_invalid = torch.FloatTensor([[0.8, 0.6],
                                            [0, 1]])

        self.assertTrue(self.sip._valid_concrete_bounds(bounds_valid))
        self.assertTrue(self.sip._valid_concrete_bounds(bounds_round))
        self.assertFalse(self.sip._valid_concrete_bounds(bounds_invalid))
