
"""
A class for loading neural networks in onnx format and converting to torch.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from typing import Optional

import torch
import torch.nn as nn
import onnx
import onnx.numpy_helper

from src.neural_networks.verinet_nn import VeriNetNN
from src.util.logger import get_logger
from src.util.config import *

logger = get_logger(LOGS_LEVEL, __name__, "../../logs/", "verifier_log")


class ONNXParser:

    def __init__(self, filepath: str):

        self.model = onnx.load(filepath)
        self.torch_model = None

    def to_pytorch(self) -> VeriNetNN:

        """
        Converts the self.onnx model to a VeriNetNN(torch) model.

        Returns:
            The VeriNetNN model.
        """

        nodes = self.model.graph.node

        curr_input_idx = nodes[0].input[0]
        mappings = []

        for node in nodes:

            mapping, curr_input_idx = self._process_node(curr_input_idx, node)

            if mapping is not None:
                mappings.append(mapping)

        self.torch_model = VeriNetNN(mappings)

        return self.torch_model

    def _process_node(self, curr_input_idx: int, node: onnx.NodeProto) -> torch.nn:

        """
        Processes a onnx node converting it to a corresponding torch node.

        Args:
            curr_input_idx:
                The expected onnx input index to the current node
            node:
                The onnx node
        Returns:
                The corresponding torch.nn operation
        """

        if node.op_type == "Relu":

            if curr_input_idx != node.input[0]:
                logger.warning(f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
            if len(node.input) != 1:
                logger.warning(f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            curr_input_idx = node.output[0]

            return nn.ReLU(), curr_input_idx

        elif node.op_type == "Sigmoid":

            if curr_input_idx != node.input[0]:
                logger.warning(f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
            if len(node.input) != 1:
                logger.warning(f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            curr_input_idx = node.output[0]

            return nn.Sigmoid(), curr_input_idx

        elif node.op_type == "Tanh":

            if curr_input_idx != node.input[0]:
                logger.warning(f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
            if len(node.input) != 1:
                logger.warning(f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            curr_input_idx = node.output[0]

            return nn.Tanh(), curr_input_idx

        elif node.op_type in ["Flatten", "Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape"]:

            # Reshape operations are assumed to adhere to the standard used in VeriNetNN and thus skipped.
            curr_input_idx = node.output[0]

            logger.info(f"Skipped node of type: {node.op_type}")

            return None, curr_input_idx

        elif node.op_type == "Gemm":

            if curr_input_idx != node.input[0]:
                logger.warning(f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
            if len(node.input) != 3:
                logger.warning(f"Unexpected input length: \n {node}, expected {3}, got {len(node.input)}")
            curr_input_idx = node.output[0]

            return self.gemm_to_torch(node), curr_input_idx

        elif node.op_type == "Conv":
            if curr_input_idx != node.input[0]:
                logger.warning(f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
            if len(node.input) != 3:
                logger.warning(f"Unexpected input length: \n {node}, expected {3}, got {len(node.input)}")
            curr_input_idx = node.output[0]

            return self.conv_to_torch(node), curr_input_idx

        else:
            logger.warning(f"Node not recognised: \n{node}")
            return None, curr_input_idx

    # noinspection PyArgumentList
    def gemm_to_torch(self, node) -> nn.Linear:

        """
        Converts a onnx 'gemm' node to a torch Linear.

        Args:
            node:
                The Gemm node.
        Returns:
            The torch Linear layer.
        """

        [weights] = [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name == node.input[2]]

        affine = nn.Linear(weights.shape[0], weights.shape[1])
        affine.weight.data = torch.Tensor(weights.copy())
        affine.bias.data = torch.Tensor(bias.copy())

        return affine

    # noinspection PyArgumentList
    def conv_to_torch(self, node) -> nn.Linear:

        """
        Converts a onnx 'Conv' node to a torch Conv.

        Args:
            node:
                The Conv node.
        Returns:
            The torch Conv layer.
        """

        [weights] = [onnx.numpy_helper.to_array(t).astype(float) for t in self.model.graph.initializer if
                     t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t).astype(float) for t in self.model.graph.initializer if
                  t.name == node.input[2]]

        dilations = 1
        groups = 1
        pads = None
        strides = 1

        for att in node.attribute:
            if att.name == "dilations":
                dilations = [i for i in att.ints]
            elif att.name == "group":
                groups = att.i
            elif att.name == "pads":
                pads = [i for i in att.ints]
            elif att.name == "strides":
                strides = [i for i in att.ints]

        conv = nn.Conv2d(weights.shape[1], weights.shape[0], weights.shape[2:4], stride=strides, padding=pads[0:2],
                         groups=groups, dilation=dilations)
        conv.weight.data = torch.Tensor(weights.copy())
        conv.bias.data = torch.Tensor(bias.copy())

        return conv


if __name__ == '__main__':

    onnx_parser = ONNXParser("../../resources/onnx/conv/cifar10_8_255.onnx")
    model = onnx_parser.to_pytorch()
