#!/usr/bin/env python
import argparse
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from typing import List, Optional


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nnet_network", type=Path, help="path to the NNET network to convert"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="path to save the ONNX model",
    )
    parser.add_argument(
        "--drop_normalization",
        action="store_true",
        help="do not include any input normalization in the converted model",
    )
    return parser.parse_args()


def build_normalize(
    means: np.ndarray,
    ranges: np.ndarray,
    input_shape: List[int],
    output_shape: List[int],
) -> nn.Module:
    flat_input_size = np.product(input_shape)
    output_shape.extend(input_shape)

    if means.shape == (1,):
        means = np.ones(input_shape) * means
        ranges = np.ones(input_shape) * ranges

    W = np.diag(1.0 / ranges[:flat_input_size])
    b = -means[:flat_input_size] / ranges[:flat_input_size]

    norm_layer = nn.Linear(flat_input_size, flat_input_size)
    norm_layer.weight.data = torch.from_numpy(W).float()
    norm_layer.bias.data = torch.from_numpy(b).float()

    return norm_layer


def build_linear(
    weights: np.ndarray,
    bias: np.ndarray,
    activation: str,
    input_shape: List[int],
    output_shape: List[int],
) -> nn.Module:
    flat_input_size = np.product(input_shape)
    output_shape.append(bias.shape[0])
    flat_output_size = np.product(output_shape)

    linear_layer = nn.Linear(flat_input_size, flat_output_size)
    linear_layer.weight.data = torch.from_numpy(weights).float()
    linear_layer.bias.data = torch.from_numpy(bias).float()

    activation_layer: Optional[nn.Module] = None
    if activation == "relu":
        activation_layer = nn.ReLU()
    elif activation == "affine":
        return linear_layer
    else:
        raise ValueError(f"Unknown activation type: {activation}")
    return nn.Sequential(linear_layer, activation_layer)


def next_line(network_file):
    while True:
        line = network_file.readline().strip().lower().strip(",")
        if line.startswith("//"):
            continue
        return line


def main(args: argparse.Namespace):
    layers: List[nn.Module] = []
    with open(args.nnet_network) as network_file:
        num_layers, input_size, output_size, max_layer_size = [
            int(v) for v in next_line(network_file).split(",")
        ]
        layer_sizes = [int(v) for v in next_line(network_file).split(",")]
        _ = next_line(network_file)
        mins = np.array([float(v) for v in next_line(network_file).split(",")])
        maxs = np.array([float(v) for v in next_line(network_file).split(",")])
        means = np.array([float(v) for v in next_line(network_file).split(",")])
        ranges = np.array([float(v) for v in next_line(network_file).split(",")])

        input_shape = [1, input_size]
        output_shape: List[int] = []
        if not args.drop_normalization:
            layers.append(build_normalize(means, ranges, input_shape, output_shape))
            input_shape = output_shape
        layer = 1
        while layer <= num_layers:
            output_shape = []
            weights = []
            for _ in range(layer_sizes[layer]):
                weights.append([float(v) for v in next_line(network_file).split(",")])
            bias = []
            for _ in range(layer_sizes[layer]):
                bias.append(float(next_line(network_file)))
            W = np.array(weights)
            b = np.array(bias)
            activation = "relu"
            if layer == num_layers:
                activation = "affine"
            layers.append(
                build_linear(
                    W,
                    b,
                    activation,
                    input_shape,
                    output_shape,
                )
            )
            input_shape = output_shape
            layer += 1
    if not args.drop_normalization:
        assert len(means) - input_size == 1
        output_shape = []
        layers.append(
            build_normalize(
                means[input_size:], ranges[input_size:], input_shape, output_shape
            )
        )
        input_shape = output_shape
    pytorch_model = nn.Sequential(*layers)
    print(pytorch_model)
    dummy_input = torch.ones([1, input_size])
    torch.onnx.export(pytorch_model, dummy_input, args.output)


if __name__ == "__main__":
    main(_parse_args())
