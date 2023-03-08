
"""
A class for loading neural networks in nnet format, as described in the readme.
The nnet format used is a slightly modified version of the ACAS Xu format (https://github.com/sisl/NNet).

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numpy as np
import torch
import torch.nn as nn

from src.neural_networks.verinet_nn import VeriNetNN


class NNET:

    """
    A class for loading .nnet.

    The class is used to convert .nnet to VeriNetNN(torch.nn.Module) objects and to normalize inputs
    """

    def __init__(self, path: str=None):

        """
        Args:
            path    : The path of the .nnet file, if given the information is read from this file. If None
                      init_nnet_from_file() or init_nnet_from_verinet_nn() can be used later.
        """

        self._main_info = None
        self._min_values = None
        self._max_values = None
        self._mean = None
        self._range = None
        self._layer_activations = None
        self._layer_types = None
        self._layer_sizes = None
        self._params = None
        self._weights = None
        self._biases = None

        self.activation_map_torch = {-1: None,
                                     0: nn.ReLU(),
                                     1: nn.Sigmoid(),
                                     2: nn.Tanh()}

        if path is not None:
            self.init_nnet_from_file(path)

    @property
    def num_inputs(self):
        return self._main_info["num_inputs"]

    @property
    def num_layers(self):
        return self._main_info["num_layers"]

    @property
    def num_outputs(self):
        return self._main_info["num_outputs"]

    @property
    def max_layer_sie(self):
        return self._main_info["max_layer_size"]

    @property
    def min_values(self):
        return self._min_values

    @property
    def max_values(self):
        return self._max_values

    @property
    def mean(self):
        return self._mean

    @property
    def range(self):
        return self._range

    @property
    def layer_activations(self):
        return self._layer_activations

    @property
    def layer_types(self):
        return self._layer_types

    @property
    def params(self):
        return self._params

    @property
    def layer_sizes(self):
        return self._layer_sizes

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    def init_nnet_from_file(self, path: str):

        """
        Reads a nnet file and stores the parameters

        Args:
            path:   The open file
        Returns:
            (layer_types, layer_size, conv_params) as lists
        """

        with open(path, "r") as f:

            last_pos = f.tell()
            line = f.readline()
            while line[0:2] == "//":  # Skip initial comments
                last_pos = f.tell()
                line = f.readline()
            f.seek(last_pos)

            # Read header
            self._read_main_info(f)
            self._read_nnet_layer_sizes(f)
            self._read_nnet_min_values(f)
            self._read_nnet_max_values(f)
            self._read_nnet_mean(f)
            self._read_nnet_range(f)
            self._read_nnet_layer_activations(f)
            self._read_nnet_layer_types(f)
            self._read_nnet_params(f)

            self._weights = []
            self._biases = []

            for layer_num, layer_type in enumerate(self.layer_types):

                if layer_type == 0:
                    self._read_nnet_fc(f, self.layer_sizes[layer_num], self.layer_sizes[layer_num + 1])

                elif layer_type == 1:
                    self._read_nnet_conv2d(f, self._params[layer_num])

                elif layer_type == 2:
                    self._read_nnet_batchnorm_2d(f, self.params[layer_num]["num_features"])

                else:
                    raise ValueError(f"Layer type: {layer_type} not recognized")

    def _read_main_info(self, file):

        """
        Reads the first line of the header and stores it in self.main_info

        Args:
            file:  The open file
        """

        line = file.readline().split(",")[:-1]

        assert len(line) == 4, f"Expected first line to have 4 ints, number of: layers, inputs, outputs, max layer " +\
                               f"size, found {len(line)}"

        self._main_info = {
                     "num_layers": int(line[0]),
                     "num_inputs": int(line[1]),
                     "num_outputs": int(line[2]),
                     "max_layer_size": int(line[3])
                     }

    def _read_nnet_layer_sizes(self, file):

        """
        Reads the second header line containing the layer sizes

        Args:
            file:  The open file
        """

        line = file.readline().split(",")[:-1]
        msg = f"File had {len(line)} layer sizes, expected {self.num_layers + 1}"
        assert len(line) == (self.num_layers + 1), msg

        self._layer_sizes = [int(size) for size in line]

    def _read_nnet_min_values(self, file):

        """
        Reads the third header line containing the minimum input values.

        Args:
            file:  The open file
        """

        line = file.readline().split(",")[:-1]

        msg = f"Got {len(line)} min values, expected 1 or {self.num_inputs}"
        assert len(line) == 1 or len(line) == self.num_inputs, msg

        self._min_values = np.array([float(min_val) for min_val in line])

    def _read_nnet_max_values(self, file):

        """
        Reads the forth header line containing the maximum input values.

        Args:
            file:  The open file
        """

        line = file.readline().split(",")[:-1]

        msg = f"Got {len(line)} max values, expected 1 or {self.num_inputs}"
        assert len(line) == 1 or len(line) == self.num_inputs, msg

        self._max_values = np.array([float(max_val) for max_val in line])

    def _read_nnet_mean(self, file):

        """
        Reads the fifth header line containing the mean input values.

        Args:
            file:  The open file
        """

        line = file.readline().split(",")[:-1]

        msg = f"Got {len(line)} mean values, expected 1, {self.num_inputs}"
        assert len(line) == 1 or len(line) == self.num_inputs, msg

        self._mean = np.array([float(mean) for mean in line])

    def _read_nnet_range(self, file):

        """
        Reads the sixth line containing the range of the input values.

        Args:
            file:  The open file
        """

        line = file.readline().split(",")[:-1]

        msg = f"Got {len(line)} range values, expected 1, {self.num_inputs}"
        assert len(line) == 1 or len(line) == self.num_inputs, msg

        self._range = np.array([float(val) for val in line])

    def _read_nnet_layer_activations(self, file):

        """
        Reads the seventh header line containing the layer activations.

        Args:
            file:  The open file
        """

        line = file.readline().split(",")[:-1]

        msg = f"Got {len(line)} activations, expected {self.num_layers}"
        assert len(line) == self.num_layers, msg

        self._layer_activations = [int(layer_type) for layer_type in line]

        for activation in self._layer_activations:
            msg = "NNET only supports Relu (0), Sigmoid (1) and Tanh (2), got activation: {activation}"
            assert activation in [-1, 0, 1, 2], msg

    def _read_nnet_layer_types(self, file):

        """
        Reads the eight header line containing the layer types.

        Args:
            file:  The open file
        """

        line = file.readline().split(",")[:-1]

        msg = f"Got {len(line)} layer types, expected {self.num_layers}"
        assert len(line) == self.num_layers, msg

        self._layer_types = [int(layer_type) for layer_type in line]

        for layer_type in self._layer_types:
            assert layer_type in [0, 1, 2], (f"NNET only supports FC (0), Conv2d (1) and BatchNorm2d (2), " +
                                             "got type {layer_type}")

    def _read_nnet_params(self, file):

        """
        Reads the ninth header line containing the convolutional parameters.

        Args:
            file:  The open file
        """

        self._params = []

        for layer_type in self._layer_types:

            params = {}

            if layer_type == 1:

                line = file.readline().split(",")[:-1]

                msg = f"Got {len(line)} convolutional parameters, expected 5"
                assert len(line) == 5, msg

                params = {"out_channels": int(line[0]), "in_channels": int(line[1]), "kernel_size": int(line[2]),
                          "stride": int(line[3]), "padding": int(line[4])}

            if layer_type == 2:

                line = file.readline().split(",")[0]
                num_features = int(line)
                line = file.readline().split(",")[:-1]
                running_mean = [np.float64(mean) for mean in line]
                line = file.readline().split(",")[:-1]
                running_var = [np.float64(var) for var in line]

                params = {"num_features": num_features, "running_mean": running_mean, "running_var": running_var}

            self._params.append(params)

    def _read_nnet_fc(self, file, in_size: int, out_size: int):

        """
        Reads and returns one fc layer of the nnet file.

        Args:
            file    : The file io stream, f calling f.readline() is assumed to return the first row of weights.
            in_size : The input size to the fc layer
            out_size: The out size of the fc layer
        """

        weights = np.empty((out_size, in_size))
        bias = np.empty(out_size)

        for node in range(out_size):
            line = file.readline().split(",")[:-1]
            weights[node, :] = np.array([float(num) for num in line])

        for node in range(out_size):
            line = file.readline().split(",")[:-1]
            bias[node] = np.array([float(num) for num in line])

        self.weights.append(weights)
        self.biases.append(bias)

    def _read_nnet_conv2d(self, file, conv_params: dict):

        """
        Reads and returns one conv2d layer of the nnet file

        Args:
            file        : The file io stream, f calling f.readline() is assumed to return the first row of weights.
            conv_params : A dict containing the params of the convo layer:
                          {'in_channels': int, 'out_channels': int, 'kernel_size': int, 'stride': int, 'padding': int}
        """

        weights = np.empty((conv_params["out_channels"], conv_params["in_channels"], conv_params["kernel_size"],
                            conv_params["kernel_size"]))
        bias = np.empty(conv_params["out_channels"])

        for channel in range(conv_params["out_channels"]):
            line = file.readline().split(",")[:-1]
            weights[channel] = np.array([np.float64(num) for num in line]).reshape((conv_params["in_channels"],
                                                                                    conv_params["kernel_size"],
                                                                                    conv_params["kernel_size"]))

        for channel in range(conv_params["out_channels"]):
            line = file.readline().split(",")[:-1]
            bias[channel] = np.array([np.float64(num) for num in line])

        self.weights.append(weights)
        self.biases.append(bias)

    def _read_nnet_batchnorm_2d(self, file, feature_num: int):

        """
        Reads and returns one batchnorm layer of the nnet file.

        Args:
            file    : The file io stream, f calling f.readline() should return the first row of weights.
        """

        weights = np.empty(feature_num)
        bias = np.empty(feature_num)

        line = file.readline().split(",")[:-1]
        weights[:] = np.array([float(num) for num in line])

        for node in range(feature_num):
            line = file.readline().split(",")[:-1]
            bias[node] = np.array([float(num) for num in line])

        self.weights.append(weights)
        self.biases.append(bias)

    def init_nnet_from_verinet_nn(self, model: VeriNetNN, input_shape: np.array, min_values: np.array,
                                  max_values: np.array, input_mean: np.array, input_range: np.array):

        """
        Gets the nnet parameters from the given model and args

        Args:
            model       : The VeriNetNN model
            input_shape : The shape of the input, either a 1d (size) or 3d (channels, height, width) array
            min_values  : The minimum values for the input, either a array of size 1 or a array of the same size as
                          the input
            max_values  : The maximum values for the input, either a array of size 1 or a array of the same size as
                          the input
            input_mean  : The mean of the inputs, either a array of size 1 or a array of the same size as
                          the input
            input_range : The range of the inputs, either a array of size 1 or a array of the same size as
                          the input
        """

        input_shape = np.array(input_shape)
        self._get_verinet_nn_layer_info(model, input_shape)
        self._get_verinet_nn_layer_activations(model)

        self._main_info = {
                     "num_layers": len(self._layer_types),
                     "num_inputs": self._layer_sizes[0],
                     "num_outputs": self._layer_sizes[-1],
                     "max_layer_size": max(self._layer_sizes)
                     }

        num_inputs = self._layer_sizes[0]

        min_values = np.atleast_1d(min_values)
        max_values = np.atleast_1d(max_values)
        input_mean = np.atleast_1d(input_mean)
        input_range = np.atleast_1d(input_range)

        msg = "min_values, max_values, input_mean and input_range should be of size 1 or the same size as the input"
        assert min_values.shape == (1,) or min_values.shape == (num_inputs, ), msg
        assert max_values.shape == (1,) or max_values.shape == (num_inputs,), msg
        assert input_mean.shape == (1,) or input_mean.shape == (num_inputs,), msg
        assert input_range.shape == (1,) or input_range.shape == (num_inputs,), msg

        self._min_values = min_values
        self._max_values = max_values
        self._mean = input_mean
        self._range = input_range

    def _get_verinet_nn_layer_info(self, model: VeriNetNN, input_shape: np.array):

        """
        Gets the layer sizes and types from the VerNetNN model

        Args:
            model       : The VeriNetNN model
            input_shape : The shape of the input, either 1d or 3d
        """

        layers = [sequential[0] for sequential in list(model.children())[0]]

        self._layer_sizes = []
        self._layer_types = []
        self._params = []
        self._weights = []
        self._biases = []
        layer_shapes = []

        self._layer_sizes.append(np.prod(input_shape))
        layer_shapes.append(input_shape)

        for i, layer in enumerate(layers):

            self._weights.append(layer.weight.data.detach().numpy())
            self._biases.append(layer.bias.data.detach().numpy())

            if isinstance(layer, nn.Linear):

                self._layer_sizes.append(layer.out_features)
                layer_shapes.append(np.array(self._layer_sizes[-1]))
                self._layer_types.append(0)

            elif isinstance(layer, nn.Conv2d):

                assert len(layer_shapes[-1]) == 3, f"Layer {i} was Conv2d, but shape of last layer was not 3"

                kernel_size = np.array(layer.kernel_size)
                padding = np.array(layer.padding)
                stride = np.array(layer.stride)

                assert kernel_size[0] == kernel_size[1], "Only square kernels are supported by nnet"
                assert padding[0] == padding[1], "Only equal padding, vertical and horizontal, is supported by nnet"
                assert stride[0] == stride[1], "Only equal stride, vertical and horizontal, is supported by nnet"

                img_size = (layer_shapes[-1][1:] + 2*padding - kernel_size) / stride + 1

                layer_shapes.append(np.array((layer.out_channels, *img_size), dtype=int))
                self._layer_sizes.append(np.prod(layer_shapes[-1]))
                self._layer_types.append(1)
                self._params.append({"out_channels": layer.out_channels,
                                     "in_channels": layer_shapes[-2][0],
                                     "kernel_size": kernel_size[0],
                                     "stride": stride[0],
                                     "padding": padding[0]})

            elif isinstance(layer, nn.BatchNorm2d):
                self._layer_sizes.append(self._layer_sizes[-1])
                layer_shapes.append(layer_shapes[-1])
                self._layer_types.append(2)
                self._params.append({"num_features": layer.num_features,
                                     "running_mean": layer.running_mean.detach().numpy(),
                                     "running_var": layer.running_var.detach().numpy()})

    def _get_verinet_nn_layer_activations(self, model: VeriNetNN):

        """
        Gets the activation functions from the VeriNetNN model

        Args:
            model:  The VeriNetNN model
        """

        sequentials = [sequential for sequential in list(model.children())[0]]
        self._layer_activations = []

        for sequential in sequentials:

            if len(list(sequential)) == 1:
                self.layer_activations.append(-1)

            elif isinstance(sequential[1], nn.ReLU):
                self.layer_activations.append(0)

            elif isinstance(sequential[1], nn.Sigmoid):
                self.layer_activations.append(1)

            elif isinstance(sequential[1], nn.Tanh):
                self.layer_activations.append(2)

            else:
                msg = f"Activation function {sequential[1]} not recognized, should be nn.Relu, nn.Sigmoid or nn.Tanh"
                raise ValueError(msg)

    # noinspection PyArgumentList
    def from_nnet_to_verinet_nn(self) -> VeriNetNN:

        """
        Converts the nnet to a VeriNet(torch.nn.Module) object.

        Returns:
            The VeriNetNN model
        """

        layers = []

        for layer_num in range(self.num_layers):

            act_num = self.layer_activations[layer_num]
            try:
                act = self.activation_map_torch[act_num]
            except KeyError:
                raise AssertionError(f"Didn't recognize activation function {act_num} for layer {layer_num}")

            layer_type_num = self.layer_types[layer_num]

            if layer_type_num == 0:
                layer = nn.Linear(self.layer_sizes[layer_num], self.layer_sizes[layer_num + 1])
                layer.weight.data = torch.Tensor(self.weights[layer_num])
                layer.bias.data = torch.Tensor(self.biases[layer_num])

            elif layer_type_num == 1:
                params = self.params[layer_num]
                layer = nn.Conv2d(params["in_channels"], params["out_channels"], params["kernel_size"],
                                  params["stride"], params["padding"])
                layer.weight.data = torch.Tensor(self.weights[layer_num])
                layer.bias.data = torch.Tensor(self.biases[layer_num])

            elif layer_type_num == 2:

                params = self.params[layer_num]
                num_features = params["num_features"]

                layer = nn.BatchNorm2d(num_features=num_features)
                layer.running_mean = torch.Tensor(params["running_mean"])
                layer.running_var = torch.Tensor(params["running_var"])
                layer.weight.data = torch.FloatTensor(self.weights[layer_num])
                layer.bias.data = torch.FloatTensor(self.biases[layer_num])

            else:
                raise AssertionError(f"Didn't recognize layer type {layer_type_num} for layer {layer_num}")

            if act is not None:
                layers.append(nn.Sequential(layer, act))
            else:
                layers.append(nn.Sequential(layer))

        return VeriNetNN(layers)

    def write_nnet_to_file(self, filepath: str):

        with open(filepath, "w") as f:

            f.write("// A neural network in nnet format\n")
            f.write("// The documentation can be found in the src/data_loader/readme.md file of the Verinet project\n")

            self._write_main_info(f)
            self._write_layer_size(f)
            self._write_min_max(f)
            self._write_mean_range(f)
            self._write_layer_activations(f)
            self._write_layer_types(f)
            self._write_layer_params(f)
            self._write_layer_weights_biases(f)

    def _write_main_info(self, f):

        """
        Writes the main info to file

        Args:
            f: The open file with write permission
        """

        num_layers = self._main_info["num_layers"]
        num_inputs = self._main_info["num_inputs"]
        num_outputs = self._main_info["num_outputs"]
        max_layer_size = self._main_info["max_layer_size"]

        f.write(f"{num_layers},{num_inputs},{num_outputs},{max_layer_size},\n")

    def _write_layer_size(self, f):

        """
        Writes the layer sizes to file

        Args:
            f: The open file with write permission
        """

        for size in self.layer_sizes:
            f.write(f"{size},")
        f.write("\n")

    def _write_min_max(self, f):

        """
        Writes the min and max values to file

        Args:
            f: The open file with write permission
        """

        for value in self.min_values:
            f.write(f"{value},")
        f.write("\n")

        for value in self.max_values:
            f.write(f"{value},")
        f.write("\n")

    def _write_mean_range(self, f):

        """
        Writes the mean and range values to file

        Args:
            f: The open file with write permission
        """

        for value in self.mean:
            f.write(f"{value},")
        f.write("\n")

        for value in self.range:
            f.write(f"{value},")
        f.write("\n")

    def _write_layer_activations(self, f):

        """
        Writes the layer activation functions to file

        Args:
            f: The open file with write permission
        """

        for act in self.layer_activations:
            f.write(f"{act},")
        f.write("\n")

    def _write_layer_types(self, f):

        """
        Writes the layer types to file

        Args:
            f: The open file with write permission
        """

        for layer_type in self.layer_types:
            f.write(f"{layer_type},")
        f.write("\n")

    def _write_layer_params(self, f):

        """
        Writes the layer parameters to file

        Args:
            f: The open file with write permission
        """

        for layer_num, layer_type in enumerate(self._layer_types):

            if layer_type == 0:
                continue

            params = self._params[layer_num]

            if layer_type == 1:

                f.write(f"{params['out_channels']},")
                f.write(f"{params['in_channels']},")
                f.write(f"{params['kernel_size']},")
                f.write(f"{params['stride']},")
                f.write(f"{params['padding']},")
                f.write("\n")

            elif layer_type == 2:

                f.write(f"{params['num_features']},\n")

                for mean in params['running_mean']:
                    f.write(f"{mean},")
                f.write("\n")
                for var in params['running_var']:
                    f.write(f"{var},")
                f.write("\n")

    def _write_layer_weights_biases(self, f):

        """
        Writes the layer weights and biases to file

        Args:
            f: The open file with write permission
        """
        for layer_num in range(len(self.layer_types)):

            weights = self._weights[layer_num]
            biases = self._biases[layer_num]

            if self._layer_types[layer_num] == 0:

                for row in weights:
                    for weight in row:
                        f.write(f"{weight},")
                    f.write("\n")

                for bias in biases:
                    f.write(f"{bias}, \n")

            if self._layer_types[layer_num] == 1:

                for out_channel in weights:
                    for in_channel in out_channel:
                        for row in in_channel:
                            for weight in row:
                                f.write(f"{weight},")
                    f.write("\n")

                for bias in biases:
                    f.write(f"{bias}, \n")

            if self._layer_types[layer_num] == 2:

                for weight in weights:
                    f.write(f"{weight},")
                f.write("\n")

                for bias in biases:
                    f.write(f"{bias}, \n")

    def normalize_input(self, x: np.array) -> np.array:

        """
        Uses the range, mean, max_values and min_values read from the nnet file to normalize the given input

        Args:
            x: The input, should be either 1D or a 2D batch, with of size: (batch_size, input_dim)
        Returns:
            The normalized input
        """

        x = self._clip_min(x)
        x = self._clip_max(x)
        x = self._subtract_mean(x)
        x = self._divide_range(x)

        return x

    def _clip_min(self, x: np.array):

        """
        Clips the input to the stored min values

        Args:
            x: The input, should be either 1D or a 2D batch, with of size: (batch_size, input_dim)
        Returns:
            The clipped input
        """

        x = x.copy()

        if (self.min_values.shape[0] == 1) or (len(x.shape) == 1 and x.shape[0] == self.num_inputs):
            x[x < self.min_values] = self.min_values

        elif (len(x.shape) == 2) and x.shape[1] == self.num_inputs:

            for row in range(x.shape[0]):
                x[row, :][x[row, :] < self.min_values] = self.min_values[x[row, :] < self.min_values]

        else:
            raise ValueError(f"Expected input to be 1D or 2D with size (batch_size, input_dim), is {x.shape}")

        return x

    def _clip_max(self, x: np.array):

        """
        Clips the input to the stored max values

        Args:
            x: The input, should be either 1D or a 2D batch, with of size: (batch_size, input_dim)
        Returns:
            The clipped input
        """

        x = x.copy()

        if (self.max_values.shape[0] == 1) or (len(x.shape) == 1 and x.shape[0] == self.num_inputs):
            x[x > self.max_values] = self.max_values

        elif (len(x.shape) == 2) and x.shape[1] == self.num_inputs:

            for row in range(x.shape[0]):
                x[row, :][x[row, :] > self.max_values] = self.max_values[x[row, :] > self.max_values]

        else:
            raise ValueError(f"Expected input to be 1D or 2D with size (batch_size, input_dim), is {x.shape}")

        return x

    def _subtract_mean(self, x: np.array):

        """
        Subtracts the mean

        Args:
            x: The input, should be either 1D or a 2D batch, with of size: (batch_size, input_dim)
        Returns:
            x-self.mean
        """

        x = x.copy()

        if (self.max_values.shape[0] == 1) or (len(x.shape) == 1 and x.shape[0] == self.num_inputs):

            x -= self._mean

        elif (len(x.shape) == 2) and x.shape[1] == self.num_inputs:

            for row in range(x.shape[0]):
                x[row, :] -= self.mean

        else:
            raise ValueError(f"Expected input to be 1D or 2D with size (batch_size, input_dim), is {x.shape}")

        return x

    def _divide_range(self, x: np.array):

        """
        Subtracts the mean

        Args:
            x: The input, should be either 1D or a 2D batch, with of size: (batch_size, input_dim)
        Returns:
            x-self.range
        """

        x = x.copy()

        if (self.max_values.shape[0] == 1) or (len(x.shape) == 1 and x.shape[0] == self.num_inputs):

            x /= self._range

        elif (len(x.shape) == 2) and x.shape[1] == self.num_inputs:

            for row in range(x.shape[0]):
                x[row, :] /= self._range

        else:
            raise ValueError(f"Expected input to be 1D or 2D with size (batch_size, input_dim), is {x.shape}")

        return x
