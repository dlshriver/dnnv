"""
A class for loading neural networks in onnx format and converting to torch.

OBS:
This code only supports ONNX _model as created in the 'save' method of VeriNetNN.
Other architectures/activations/computational graphs are not considered and will most
likely fail.

Author: Patrick Henriksen <patrick@henriksen.as>
"""


from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnx.numpy_helper

from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode
from verinet.neural_networks.custom_layers import Mean, MulConstant, AddDynamic, Reshape, Transpose, AddConstant, \
    Unsqueeze
from verinet.util.logger import get_logger
from verinet.util.config import CONFIG

logger = get_logger(CONFIG.LOGS_LEVEL, __name__, "../../logs/", "log")


class ONNXParser:

    # noinspection PyUnresolvedReferences
    def __init__(self,
                 filepath: str,
                 transpose_fc_weights: bool = False,
                 input_names: tuple = ('0', 'x', 'x:0', 'X_0', 'input', 'input.1', 'Input_1', 'ImageInputLayer'),
                 use_64bit: bool = False):

        """
        Args:
            filepath:
                The path of the onnx file
            transpose_fc_weights:
                If true, weights are transposed for fully-connected layers.
            input_names:
                The name of the network's input in the onnx _model.
            use_64bit:
                If true, values are stored as 64 bit.
        """

        try:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import dnnv
            op_graph = dnnv.nn.parse(Path(filepath)).simplify()
            self._model = op_graph.as_onnx()
        except ModuleNotFoundError:
            logger.warning("DNNV package not found, attempting to proceed without network simplification.")
            self._model = onnx.load(filepath)
        except ImportError:
            logger.warning("DNNV package not found, attempting to proceed without network simplification.")
            self._model = onnx.load(filepath)
        except ValueError as e:
            logger.info(f"DNNV simplification failed with message:\n'{str(e)}'\n"
                        f"Attempting to proceed without network simplification.")
            self._model = onnx.load(filepath)

        self._input_names = input_names
        self._transpose_weights = transpose_fc_weights

        self.torch_model = None

        self._pad_nodes = []
        self._constant_nodes = []
        self._bias_nodes = []
        self._onnx_nodes = None

        self._infered_shapes = onnx.shape_inference.infer_shapes(self._model).graph.value_info
        self._tensor_type = torch.DoubleTensor if use_64bit else torch.FloatTensor

        self._node_to_idx = {}

    def to_pytorch(self) -> VeriNetNN:

        """
        Converts the self.onnx _model to a VeriNetNN(torch) _model.

        Returns:
            The VeriNetNN _model.
        """

        self._onnx_nodes = list(self._model.graph.node)
        self._onnx_nodes = self._simplify_complex_flatten(self._onnx_nodes)
        self._onnx_nodes = self._filter_special_nodes(self._onnx_nodes)
        self._onnx_nodes = self._filter_mul_nodes(self._onnx_nodes)
        verinet_nn_nodes = self._process_all_nodes(onnx_nodes=self._onnx_nodes)

        last_idx = verinet_nn_nodes[-1].idx
        verinet_nn_nodes.append(VeriNetNNNode(last_idx+1, nn.Identity(), [last_idx], []))
        verinet_nn_nodes[-2].connections_to = [last_idx+1]

        if len(self._pad_nodes) != 0:
            logger.warning(f"Model contained unprocessed padding nodes")

        if len(self._bias_nodes) != 0:
            logger.warning(f"Model contained unprocessed bias nodes")

        for node in self._constant_nodes:
            value = float(onnx.numpy_helper.to_array(node.attribute[0].t))
            if value != 1:
                logger.warning(f"Model contained unprocessed constant nodes")
                break

        return VeriNetNN(verinet_nn_nodes)

    def _simplify_complex_flatten(self, onnx_nodes: list) -> list:

        """
        Simplifies operational chains of Gather, Unsqueeze, Concat, Reshape to
        flatten when possible.

        Args:
            onnx_nodes:
                A list of the onnx nodes.
        Returns:
            A list of the simplified onnx nodes.
        """

        remove_nodes = []
        insert_nodes = []
        num_flatt = 0

        for i, shape_node in enumerate(self._onnx_nodes):
            if not shape_node.op_type == "Shape" or len(shape_node.output) != 1:
                continue

            gather_nodes = [node for node in onnx_nodes if shape_node.output[0] in node.input]
            # gather_indices = onnx.numpy_helper.to_array(node.attribute[0].t
            if (len(gather_nodes) != 1 or gather_nodes[0].attribute[0].name != "axis" or
                    gather_nodes[0].attribute[0].i != 0 or len(gather_nodes[0].output) != 1 or
                    gather_nodes[0].op_type != "Gather"):
                continue
            gather_node = gather_nodes[0]

            gather_indices_node = [node for node in onnx_nodes if gather_node.input[1] in node.output][0]
            indices = onnx.numpy_helper.to_array(gather_indices_node.attribute[0].t)
            if indices != 0:
                continue

            unsqueeze_nodes = [node for node in onnx_nodes if gather_node.output[0] in node.input]
            if (len(unsqueeze_nodes) != 1 or unsqueeze_nodes[0].attribute[0].name != "axes" or
                    unsqueeze_nodes[0].attribute[0].i != 0 or len(unsqueeze_nodes[0].output) != 1 or
                    unsqueeze_nodes[0].op_type != "Unsqueeze"):
                continue
            unsqueeze_node = unsqueeze_nodes[0]

            concat_nodes = [node for node in onnx_nodes if unsqueeze_node.output[0] in node.input]
            if (len(concat_nodes) != 1 or concat_nodes[0].attribute[0].name != "axis" or
                    concat_nodes[0].attribute[0].i != 0 or len(concat_nodes[0].output) != 1 or
                    concat_nodes[0].op_type != "Concat"):
                continue
            concat_node = concat_nodes[0]

            reshape_nodes = [node for node in onnx_nodes if concat_node.output[0] in node.input]
            if len(reshape_nodes) != 1 or reshape_nodes[0].op_type != "Reshape":
                continue
            reshape_node = reshape_nodes[0]

            input_nodes = [node for node in onnx_nodes if shape_node.input[0] in node.output]
            if len(input_nodes) != 1:
                continue
            input_node = input_nodes[0]

            insert_nodes.append((CustomNode(inputs=input_node.output, output=reshape_node.output,
                                            op_type="Flatten", name=f"CustomFlatt_{num_flatt}"), i))
            num_flatt += 1
            remove_nodes = [gather_node, unsqueeze_node, concat_node, reshape_node,
                            gather_indices_node]  # shape node is replaced below.

        for new_node, i in insert_nodes:
            self._onnx_nodes[i] = new_node

        for node in remove_nodes:
            onnx_nodes.remove(node)

        return onnx_nodes

    def _filter_special_nodes(self, onnx_nodes: list) -> list:

        """
        Filters out special nodes (constant and pad).

        Indices of other nodes input and output are adjusted accordingly.

        Args:
            onnx_nodes:
                The list of all onnx nodes
        """

        new_nodes = []

        for node in onnx_nodes:

            if node.op_type == "Constant":
                self._constant_nodes.append(node)
            elif node.op_type == "Pad":

                self._pad_nodes.append(node)

                if len(node.input) != 1:
                    raise ValueError(f"Expected input of len 1 for: {node}")

            elif node.op_type == "Add":  # Filter all add-nodes that are biases for MatMul nodes

                connected_1 = [other_node for other_node in self._onnx_nodes if node.input[0] in other_node.output]
                connected_2 = [other_node for other_node in self._onnx_nodes if node.input[1] in other_node.output]

                if len(connected_1) == 1 and connected_1[0].op_type == "MatMul" and len(connected_2) == 0:
                    self._bias_nodes.append(node)

                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)

        return new_nodes

    def _filter_mul_nodes(self, onnx_nodes: list) -> list:

        """
        Filters out all mul nodes with multiplier 1.
        """

        new_nodes = []

        for node in onnx_nodes:
            if node.op_type == "Mul":

                const_nodes = [const_node for const_node in self._constant_nodes if const_node.output[0] in node.input]

                if len(const_nodes) != 1:
                    raise ValueError(f"Expected exactly one constant node, got {const_nodes}")

                const_node = const_nodes[0]

                if len(const_node.output) > 1:
                    raise ValueError(f"Expected constant node to have one output: {const_node}")

                atts = const_node.attribute

                if len(atts) > 1 or atts[0].name != "value":
                    raise ValueError(f"Expected constant a single 'value' attribute: {const_node}")

                value = float(onnx.numpy_helper.to_array(atts[0].t))

                if value == 1:
                    in_idx = node.input[0] if node.input[0] not in const_node.output else node.input[1]

                    for out_idx in node.output:
                        self._skip_node(in_idx, out_idx)

                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)

        return new_nodes

    def _process_all_nodes(self, onnx_nodes: list) -> list:

        """
        Loops through all onnx_nodes converting to the corresponding VeriNetNN nodes.

        Args:
            onnx_nodes:
                A list of all onnx nodes.
        Returns:
            A list of the VeriNetNN nodes.
        """

        self._node_to_idx = {}
        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity())]

        idx_num = 0
        for onnx_node in onnx_nodes:

            new_verinet_nn_nodes = self._process_node(onnx_node, idx_num + 1)

            if new_verinet_nn_nodes is not None:
                verinet_nn_nodes += new_verinet_nn_nodes
                idx_num += len(new_verinet_nn_nodes)
                self._node_to_idx[onnx_node.name] = idx_num

        self._add_output_connections(verinet_nn_nodes)
        return verinet_nn_nodes

    @staticmethod
    def _add_output_connections(verinet_nn_nodes: list):

        """
        Adds the output connections for all VeriNetNN nodes.

        Args:
            verinet_nn_nodes:
                The VeriNetNN nodes.
        """

        for i, node in enumerate(verinet_nn_nodes):
            for in_idx in node.connections_from:
                verinet_nn_nodes[in_idx].connections_to.append(i)

    def _process_node(self, node: onnx.NodeProto, idx_num: int) -> Optional[list]:

        """
        Processes an onnx node converting it to a corresponding torch node.

        Args:
            node:
                The onnx node
            idx_num:
                The current node-index number
        Returns:
                A list of corresponding VeriNetNN nodes.
        """

        if node.op_type == "Relu":
            input_connections = self._get_connections_to(node)
            if len(input_connections) > 1:
                raise ValueError(f"Found more than one input connection to {node}")
            return [VeriNetNNNode(idx_num, nn.ReLU(), input_connections)]

        elif node.op_type == "PRelu":
            if len(node.input) != 2:
                raise ValueError(f"Found more than one input connection to {node}")
            return self.prelu_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Sigmoid":
            input_connections = self._get_connections_to(node)
            if len(input_connections) > 1:
                raise ValueError(f"Found more than one input connection to {node}")
            return [VeriNetNNNode(idx_num, nn.Sigmoid(), input_connections)]

        elif node.op_type == "Tanh":
            input_connections = self._get_connections_to(node)
            if len(input_connections) > 1:
                raise ValueError(f"Found more than one input connection to {node}")
            return [VeriNetNNNode(idx_num, nn.Tanh(), input_connections)]

        elif node.op_type == "Flatten":
            input_connections = self._get_connections_to(node)
            if len(input_connections) > 1:
                raise ValueError(f"Found more than one input connection to {node}")
            return [VeriNetNNNode(idx_num, nn.Flatten(), input_connections)]

        elif node.op_type == "Gemm":
            if len(node.input) != 3:
                logger.warning(f"Unexpected input length: \n {node}, expected {3}, got {len(node.input)}")
            return self.gemm_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "MatMul":
            if len(node.input) != 2:
                logger.warning(f"Unexpected input length: \n{node}, expected {2}, got {len(node.input)}")
            return self.matmul_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Conv":
            if len(node.input) != 2 and len(node.input) != 3:
                logger.warning(f"Unexpected input length: \n {node}, expected {3}, got {len(node.input)}")
            return self.conv_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "BatchNormalization":
            if len(node.input) != 5:
                logger.warning(f"Unexpected input length: \n {node}, expected {5}, got {len(node.input)}")
            return self.batch_norm_2d_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "ReduceMean":
            if len(node.input) != 1:
                logger.warning(f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            return self.reduce_mean_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "AveragePool":
            if len(node.input) != 1:
                logger.warning(f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            return self.avg_pool_2d_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Mul":
            if len(node.input) != 2:
                logger.warning(f"Unexpected input length: \n{node}, expected {2}, got {len(node.input)}")
            return self.mul_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Div":
            if len(node.input) != 2:
                logger.warning(f"Unexpected input length: \n{node}, expected {2}, got {len(node.input)}")
            return self.div_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Reshape":
            if len(node.input) != 2:
                logger.warning(f"Unexpected input length: \n{node}, expected {2}, got {len(node.input)}")
            return self.reshape_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Unsqueeze":
            if len(node.input) != 1:
                logger.warning(f"Unexpected input length: \n{node}, expected {2}, got {len(node.input)}")
            return self.unsqueeze_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Add":
            if len(node.input) != 2:
                logger.warning(f"Unexpected input length: \n{node}, expected {2}, got {len(node.input)}")
            return self.add_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Sub":
            if len(node.input) != 2:
                logger.warning(f"Unexpected input length: \n{node}, expected {2}, got {len(node.input)}")
            return self.sub_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Transpose":
            if len(node.input) != 1:
                logger.warning(f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            return self.transpose_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "MaxPool":
            if len(node.input) != 1:
                logger.warning(f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            return self.maxpool_to_verinet_nn_node(node, idx_num)

        elif node.op_type in ["Softmax"]:
            if len(node.input) != 1:
                raise ValueError(f"Found unexpected node that could not be skipped: {node}")
            from_idx = node.input[0]
            for out_idx in node.output:
                self._skip_node(from_idx, out_idx)

            logger.warning(f"Skipped node of type: {node.op_type}")
            return None

        else:
            logger.warning(f"Node not recognised: \n{node}")
            return None

    # noinspection PyArgumentList,PyCallingNonCallable
    def prelu_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'gemm' node to a Linear verinet_nn_node.

        Args:
            node:
                The Gemm node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the torch Linear node.
        """

        [weights] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[1]]

        prelu = nn.PReLU(len(weights))
        prelu.weight.data = self._tensor_type(weights.copy())

        input_connections = self._get_connections_to(node)
        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, prelu, input_connections)]

    # noinspection PyArgumentList,PyCallingNonCallable
    def gemm_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'gemm' node to a Linear verinet_nn_node.

        Args:
            node:
                The Gemm node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the torch Linear node.
        """

        [weights] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[2]]

        if self._transpose_weights:
            weights = weights.T

        affine = nn.Linear(weights.shape[1], weights.shape[0])
        affine.weight.data = self._tensor_type(weights.copy())
        affine.bias.data = self._tensor_type(bias.copy())

        input_connections = self._get_connections_to(node)
        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, affine, input_connections)]

    # noinspection PyCallingNonCallable
    def matmul_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'matmul' node to a Linear verinet_nn_node.

        Args:
            node:
                The matmul node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the torch Linear node.
        """

        if len(node.output) != 1:
            raise ValueError("Expected MatMul to have 1 output")

        bias_nodes = [bias_node for bias_node in self._bias_nodes if bias_node.input[0] == node.output[0]]

        if len(bias_nodes) != 1:
            raise ValueError("Expected MatMul to have at most 1 bias node")

        bias_node = bias_nodes[0]
        self._bias_nodes.remove(bias_node)

        node.output[0] = bias_node.output[0]

        [weights] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if
                     t.name == node.input[0] or t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if
                  t.name == bias_node.input[0] or t.name == bias_node.input[1]]

        if self._transpose_weights:
            weights = weights.T

        affine = nn.Linear(weights.shape[1], weights.shape[0])
        affine.weight.data = self._tensor_type(weights.copy())
        affine.bias.data = self._tensor_type(bias.copy())

        input_connections = self._get_connections_to(node)
        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, affine, input_connections)]

    # noinspection PyArgumentList,PyCallingNonCallable,PyTypeChecker
    def conv_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'Conv' node to a Conv verinet_nn_node.

        Args:
            node:
                The Conv node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node Conv node.
        """

        [weights] = [onnx.numpy_helper.to_array(t).astype(float) for t in self._model.graph.initializer if
                     t.name == node.input[1]]

        if len(node.input) >= 3:
            [bias] = [onnx.numpy_helper.to_array(t).astype(float) for t in self._model.graph.initializer if
                      t.name == node.input[2]]
        else:
            bias = np.zeros(weights.shape[0])

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
                pads = [i for i in att.ints][0:2]
            elif att.name == "strides":
                strides = [i for i in att.ints]

        if pads is None:
            pads = 0

        conv = nn.Conv2d(weights.shape[1]*groups, weights.shape[0], weights.shape[2:4], stride=strides,
                         padding=pads, groups=groups, dilation=dilations)

        conv.weight.data = self._tensor_type(weights.copy())
        conv.bias.data = self._tensor_type(bias.copy())

        input_connections = self._get_connections_to(node)
        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, conv, input_connections)]

    # noinspection PyArgumentList,PyCallingNonCallable
    def batch_norm_2d_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'BatchNormalization' node to a BatchNorm2d verinet_nn_node.

        Args:
            node:
                The BatchNormalization node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node BatchNorm2d node.
        """

        [weights] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[2]]
        [mean] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[3]]
        [var] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[4]]

        batch_norm_2d = nn.BatchNorm2d(weights.shape[0])

        batch_norm_2d.weight.data = self._tensor_type(weights.copy())
        batch_norm_2d.bias.data = self._tensor_type(bias.copy())
        batch_norm_2d.running_mean.data = self._tensor_type(mean.copy())
        batch_norm_2d.running_var.data = self._tensor_type(var.copy())

        for att in node.attribute:
            if att.name == "epsilon":
                batch_norm_2d.eps = att.f
            elif att.name == 'momentum':
                batch_norm_2d.momentum = att.f

        input_connections = self._get_connections_to(node)
        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, batch_norm_2d, input_connections)]

    # noinspection PyArgumentList
    def reduce_mean_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'Reduce Mean' node to a Mean verinet_nn_node.

        Args:
            node:
                The ReduceMean node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node ReduceMean node.
        """

        dims = None
        keepdim = False

        for att in node.attribute:
            if att.name == "axes":
                dims = tuple(att.ints)
            elif att.name == 'keepdims':
                keepdim = att.i != 0

        input_connections = self._get_connections_to(node)
        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, Mean(dims, keepdim), input_connections)]

    # noinspection PyArgumentList
    def avg_pool_2d_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts a onnx 'AvgPool2d' node to a AvgPool2d verinet_nn_node.

        Args:
            node:
                The AvgPool2d node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node AvgPool2d node.
        """

        kernel_size = None
        stride = None
        padding = None

        for att in node.attribute:

            if att.name == "pads":
                padding = att.ints[0:2]

                if att.ints[0] != att.ints[2] or att.ints[1] != att.ints[3]:
                    logger.warning(f"Unexpected pad parameters for {att.ints}")

            elif att.name == 'kernel_shape':
                kernel_size = tuple(att.ints)
            elif att.name == 'strides':
                stride = tuple(att.ints)

        this_pad_nodes = [pad_node for pad_node in self._pad_nodes if pad_node.output == node.input]

        if len(this_pad_nodes) > 1:
            raise ValueError(f"Expected at most one pad node for AvgPool, got {this_pad_nodes}")

        if len(this_pad_nodes) == 1:
            pad_node = this_pad_nodes[0]

            for att in pad_node.attribute:
                if att.name == "pads":

                    if att.ints[2] != att.ints[6] or att.ints[3] != att.ints[7]:
                        raise ValueError(f"Unexpected pad parameters for {att.ints}")

                    padding[0] += att.ints[2]
                    padding[1] += att.ints[3]

            self._pad_nodes.remove(pad_node)

            input_connections = self._get_connections_to(pad_node)
        else:
            input_connections = self._get_connections_to(node)

        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num,
                              nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=tuple(padding)),
                              input_connections)]

    # noinspection PyCallingNonCallable
    def mul_to_verinet_nn_node(self, node, idx_num: int) -> Optional[list]:

        """
        Converts an onnx 'Mul' node to a MulConstant verinet_nn_node.

        Args:
            node:
                The Mul node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node MulConstant op.
        """

        const_nodes = [const_node for const_node in self._constant_nodes if const_node.output[0] in node.input]

        if len(const_nodes) != 1:
            raise ValueError(f"Expected exactly one constant node, got {const_nodes}")

        const_node = const_nodes[0]

        if len(const_node.output) > 1:
            raise ValueError(f"Expected constant node to have one output: {const_node}")

        atts = const_node.attribute

        if len(atts) > 1 or atts[0].name != "value":
            raise ValueError(f"Expected constant a single 'value' attribute: {const_node}")

        value = self._tensor_type(onnx.numpy_helper.to_array(atts[0].t).copy())
        self._constant_nodes.remove(const_node)

        # Skip nodes that multiply by 1.
        if torch.sum(value.reshape(-1) != 1) == 0 and len(node.output) == 1:

            [succeding_node] = [other_node for other_node in self._onnx_nodes if node.output[0] in other_node.input]

            for i in range(len(succeding_node.input)):
                if succeding_node.input[i] == node.output[0]:
                    succeding_node.input[i] = node.input[0]

            return None

        input_connections = self._get_connections_to(node)

        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, MulConstant(value), input_connections)]

    # noinspection PyCallingNonCallable
    def div_to_verinet_nn_node(self, node, idx_num: int) -> Optional[list]:

        """
        Converts an onnx 'Div' node to a MulConstant verinet_nn_node.

        Args:
            node:
                The Div node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node MulConstant op.
        """

        const_nodes = [const_node for const_node in self._constant_nodes if const_node.output[0] in node.input]

        if len(const_nodes) != 1:
            raise ValueError(f"Expected exactly one constant node, got {const_nodes}")

        const_node = const_nodes[0]

        if len(const_node.output) > 1:
            raise ValueError(f"Expected constant node to have one output: {const_node}")

        atts = const_node.attribute

        if len(atts) > 1 or atts[0].name != "value":
            raise ValueError(f"Expected constant a single 'value' attribute: {const_node}")

        value = self._tensor_type(onnx.numpy_helper.to_array(atts[0].t).copy())
        self._constant_nodes.remove(const_node)

        # Skip nodes that divide by 1.
        if torch.sum(value.reshape(-1) != 1) == 0 and len(node.output) == 1:

            [succeding_node] = [other_node for other_node in self._onnx_nodes if node.output[0] in other_node.input]

            for i in range(len(succeding_node.input)):
                if succeding_node.input[i] == node.output[0]:
                    succeding_node.input[i] = node.input[0]

            return None

        input_connections = self._get_connections_to(node)

        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, MulConstant(1/value), input_connections)]

    def reshape_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'Reshape' node to a Reshape-verinet_nn_node.

        Args:
            node:
                The Reshape-node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node Reshape op.
        """

        value = None
        const_nodes = [const_node for const_node in self._constant_nodes if const_node.output[0] in node.input]

        if len(const_nodes) > 1:
            raise ValueError(f"Expected exactly at most one constant node, got {const_nodes}")

        elif len(const_nodes) == 1:
            const_node = const_nodes[0]

            if len(const_node.output) > 1:
                raise ValueError(f"Expected constant node to have one output: {const_node}")

            atts = const_node.attribute

            if len(atts) > 1 or atts[0].name != "value":
                raise ValueError(f"Expected constant a single 'value' attribute: {const_node}")

            value = tuple(onnx.numpy_helper.to_array(atts[0].t))
            self._constant_nodes.remove(const_node)

        else:
            for init in self._model.graph.initializer:
                if init.name == node.input[1]:
                    value = tuple(onnx.numpy_helper.to_array(init))

        input_connections = self._get_connections_to(node)

        if value is None:
            raise ValueError("Could not find a shape for Reshape operation")

        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, Reshape(value), input_connections)]

    def unsqueeze_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts a onnx 'Unsqueeze' node to a Unsqueeze verinet_nn_node.

        Args:
            node:
                The Unsqueeze node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the Unsqueeze verinet_nn_node.
        """

        atts = node.attribute
        dims = tuple(atts[0].ints)

        input_connections = self._get_connections_to(node)

        return [VeriNetNNNode(idx_num, Unsqueeze(dims), input_connections)]

    def add_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'Add' node to a AddDynamic verinet_nn_node.

        The input is assumed to be the outputs of two previous layers, this
        method does not handle add operations between one layer-output and one
        constant value.

        Args:
            node:
                The Add node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node AddDynamic op.
        """

        input_connections = self._get_connections_to(node)

        if len(input_connections) != 2:
            raise ValueError(f"Expected 2 inputs for {node}, got {input_connections}")

        return [VeriNetNNNode(idx_num, AddDynamic(), input_connections)]

    # noinspection PyCallingNonCallable
    def sub_to_verinet_nn_node(self, node, idx_num: int) -> Optional[list]:

        """
        Converts an onnx 'Sub' node to a AddConstant verinet_nn_node.

        The input is assumed to be the output of one previous layers and a constant,
        this method does not handle sub operations between two layers.

        Args:
            node:
                The Sub node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node AddConstant op.
        """

        const_nodes = [const_node for const_node in self._constant_nodes if const_node.output[0] in node.input]

        if len(const_nodes) == 0:
            [value] = [self._tensor_type(onnx.numpy_helper.to_array(t).copy()) for t in self._model.graph.initializer
                       if t.name == node.input[1]]

        elif len(const_nodes) == 1:

            const_node = const_nodes[0]

            if len(const_node.output) > 1:
                raise ValueError(f"Expected constant node to have one output: {const_node}")

            atts = const_node.attribute

            if len(atts) > 1 or atts[0].name != "value":
                raise ValueError(f"Expected constant a single 'value' attribute: {const_node}")

            value = self._tensor_type(onnx.numpy_helper.to_array(atts[0].t).copy())
            self._constant_nodes.remove(const_node)
        else:
            raise ValueError(f"Expected exactly one constant node, got {const_nodes}")

        # Skip nodes that subtract by 0.
        if torch.sum(value.reshape(-1) != 0) == 0 and len(node.output) == 1:

            [succeding_node] = [other_node for other_node in self._onnx_nodes if node.output[0] in other_node.input]

            for i in range(len(succeding_node.input)):
                if succeding_node.input[i] == node.output[0]:
                    succeding_node.input[i] = node.input[0]

            return None

        input_connections = self._get_connections_to(node)

        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, AddConstant(-value), input_connections)]

    def transpose_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'Transpose' node to a Transpose verinet_nn_node.

        Args:
            node:
                The Transpose node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node Transpose op.
        """

        input_connections = self._get_connections_to(node)
        perm_dims = None

        for att in node.attribute:
            if att.name == "perm":
                perm_dims = tuple(att.ints)

        if perm_dims is None:
            raise ValueError("Could not find dimensions for transpose operation")

        return [VeriNetNNNode(idx_num, Transpose(perm_dims), input_connections)]

    # noinspection PyCallingNonCallable,PyTypeChecker
    def maxpool_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'MaxPool' node to a set of convolutional/Relu layers
        equivalent to the maxpool op.

        Args:
            node:
                The MaxPool node.
            idx_num:
                The index of the current node.
        Returns:
            The verinet_nn_node MaxPool op.
        """

        input_connections = self._get_connections_to(node)

        # MaxPool currently assumes that input is strictly positive.
        for other_node in self._onnx_nodes:
            try:
                if self._node_to_idx[other_node.name] in input_connections:
                    if other_node.op_type != "Relu":
                        raise ValueError("Expected all nodes preceding MaxPool to be Relu.")
            except KeyError:
                continue

        kernel_shape, pads, strides = None, None, None

        for att in node.attribute:

            if att.name == "kernel_shape":
                kernel_shape = np.array(att.ints)

            if att.name == "pads":
                pads = np.array(att.ints)

            if att.name == "strides":
                strides = np.array(att.ints)

        if sum(pads != 0) > 0:
            raise ValueError("Padding not implemented for MaxPool")
        if kernel_shape[0] % 2 != 0 or kernel_shape[1] % 2 != 0:
            raise ValueError("Expected kernel shape in MaxPool to be divisible by 2")
        if strides[0] != kernel_shape[0] or strides[1] != kernel_shape[1]:
            raise ValueError("Expected strides to be equal to kernels shape")

        infered_shape = [shape for shape in self._infered_shapes if shape.name == node.input[0]][0]
        shape = tuple([d.dim_value for d in infered_shape.type.tensor_type.shape.dim])

        if len(shape) != 4:
            raise ValueError("Expected MaxPool to have 4 dimensional input where the first dim is the batch.")

        num_channels = shape[1]

        # First vertical conv layer
        layers = [nn.Conv2d(in_channels=num_channels, out_channels=num_channels * 2, groups=num_channels,
                            kernel_size=(2, 1), stride=(2, 1), bias=False)]
        layers[-1].weight.data[::2] = self._tensor_type([[[1], [0]]])
        layers[-1].weight.data[1::2] = self._tensor_type([[[-1], [1]]])
        layers.append(nn.ReLU())

        # Remaining vertical
        for i in range(torch.trunc(kernel_shape[0] / 2) - 1):
            layers.append(nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels*2, groups=num_channels,
                                    kernel_size=(2, 1), stride=(2, 1), bias=False))
            layers[-1].weight.data[::2] = self._tensor_type([[[1], [0]], [[1], [0]]])
            layers[-1].weight.data[1::2] = self._tensor_type([[[-1], [1]], [[-1], [1]]])
            layers.append(nn.ReLU())

        # Collapse extra channel
        layers.append(nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels, groups=num_channels,
                                kernel_size=(1, 1), stride=(1, 1), bias=False))
        layers[-1].weight.data[:] = self._tensor_type([[[1]], [[1]]])

        # First horizontal conv layer
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=num_channels * 2, groups=num_channels,
                                kernel_size=(1, 2), stride=(1, 2), bias=False))
        layers[-1].weight.data[::2] = self._tensor_type([[[1, 0]]])
        layers[-1].weight.data[1::2] = self._tensor_type([[[-1, 1]]])
        layers.append(nn.ReLU())

        # Remaining horizontal
        for i in range(torch.trunc(kernel_shape[0] / 2) - 1):
            layers.append(nn.Conv2d(in_channels=num_channels * 2, out_channels=num_channels * 2, groups=num_channels,
                                    kernel_size=(1, 2), stride=(1, 2), bias=False))
            layers[-1].weight.data[::2] = self._tensor_type([[[1, 0]], [[1, 0]]])
            layers[-1].weight.data[1::2] = self._tensor_type([[[-1, 1]], [[-1, 1]]])
            layers.append(nn.ReLU())

        # Collapse extra channel
        layers.append(nn.Conv2d(in_channels=num_channels * 2, out_channels=num_channels, groups=num_channels,
                                kernel_size=(1, 1), stride=(1, 1), bias=False))
        layers[-1].weight.data[:] = self._tensor_type([[[1]], [[1]]])

        verinet_nn_nodes = []
        for layer in layers:
            verinet_nn_nodes.append(VeriNetNNNode(idx_num, layer, input_connections))
            input_connections = [idx_num]
            idx_num += 1

        return verinet_nn_nodes

    def _get_connections_to(self, node) -> list:

        """
        Returns the indices of all nodes that are connected to the given node.

        Note that any node with connections to the given node is assumed to already
        have been processed.

        Args:
            node:
                The onnx node.
        Returns:
            A list of indices of nodes connected to the given node.
        """

        input_idx = []

        for onnx_node in self._onnx_nodes:
            for output in onnx_node.output:
                if output in node.input:
                    input_idx.append(self._node_to_idx[onnx_node.name])

        for in_connection in node.input:
            if in_connection in self._input_names:
                input_idx.append(0)

        return input_idx

    def _skip_node(self, in_idx: int, out_idx: int):

        """
        Changes the indices of all nodes to bypass the given node.

        Args:
            in_idx:
                The in-indices of the node to bypass.
            out_idx:
                The out-indices of the node to bypass.
        """

        for other_node in self._onnx_nodes + self._pad_nodes:
            for i in range(len(other_node.input)):
                if other_node.input[i] == out_idx:
                    other_node.input[i] = in_idx


class CustomNode:

    def __init__(self, inputs: list, output: list, op_type: str, name: str):

        self.input = inputs
        self.output = output
        self.op_type = op_type
        self.name = name
