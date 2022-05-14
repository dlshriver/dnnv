from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def build_known_behavior_artifacts():
    # creates a set of networks with known behavior

    artifact_dir = Path(__file__).parent / "networks"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # create models for known functions

    const_zero_path = artifact_dir / "const_zero.onnx"
    if not const_zero_path.exists():
        dummy_input = torch.ones((1, 2))
        fc1 = nn.Linear(2, 5)
        fc1.weight.data = torch.zeros(5, 2)
        fc1.bias.data = torch.zeros(5)
        fc2 = nn.Linear(5, 1)
        fc2.weight.data = torch.zeros(1, 5)
        fc2.bias.data = torch.zeros(1)
        const_zero = nn.Sequential(fc1, nn.ReLU(), fc2).eval()
        torch.onnx.export(
            const_zero,
            dummy_input,
            const_zero_path,
            input_names=["input"],
            dynamic_axes={"input": [0]},
        )

    const_one_path = artifact_dir / "const_one.onnx"
    if not const_one_path.exists():
        dummy_input = torch.ones((1, 2))
        fc1 = nn.Linear(2, 5)
        fc1.weight.data = torch.zeros(5, 2)
        fc1.bias.data = torch.zeros(5)
        fc2 = nn.Linear(5, 1)
        fc2.weight.data = torch.zeros(1, 5)
        fc2.bias.data = torch.ones(1)
        const_one = nn.Sequential(fc1, nn.ReLU(), fc2).eval()
        torch.onnx.export(
            const_one,
            dummy_input,
            const_one_path,
            input_names=["input"],
            dynamic_axes={"input": [0]},
        )

    sum_gt_one_path = artifact_dir / "sum_gt_one.onnx"
    if not sum_gt_one_path.exists():
        dummy_input = torch.ones((1, 10))
        fc1 = nn.Linear(10, 1)
        fc1.weight.data = torch.ones(1, 10)
        fc1.bias.data = -torch.ones(1)
        fc2 = nn.Linear(1, 1)
        fc2.weight.data = torch.ones(1, 1)
        fc2.bias.data = torch.zeros(1)
        sum_gt_one = nn.Sequential(fc1, nn.ReLU(), fc2).eval()
        torch.onnx.export(
            sum_gt_one,
            dummy_input,
            sum_gt_one_path,
            input_names=["input"],
            dynamic_axes={"input": [0]},
        )

    a_gt_b_path = artifact_dir / "a_gt_b.onnx"  # class 0: a > b, class 1: b > a
    if not a_gt_b_path.exists():
        dummy_input = torch.ones((1, 2))
        fc1 = nn.Linear(2, 2)
        fc1.weight.data = torch.ones(2, 2)
        fc1.weight.data[0, 1] = -1
        fc1.weight.data[1, 0] = -1
        fc1.bias.data = torch.zeros(2)
        fc2 = nn.Linear(2, 2)
        fc2.weight.data = torch.eye(2)
        fc2.bias.data = torch.zeros(2)
        a_gt_b = nn.Sequential(fc1, nn.ReLU(), fc2, nn.Softmax(dim=1)).eval()
        torch.onnx.export(
            a_gt_b,
            dummy_input,
            a_gt_b_path,
            input_names=["input"],
            dynamic_axes={"input": [0]},
        )

    max_positive_path = artifact_dir / "max_positive.onnx"
    if not max_positive_path.exists():
        dummy_input = torch.ones((1, 3, 4, 4))
        conv1 = nn.Conv2d(3, 3, 1)
        conv1.weight.data = torch.eye(3)[..., None, None]
        conv1.bias.data = torch.zeros(3)
        max1 = nn.MaxPool2d((2, 2), (2, 2))
        conv2 = nn.Conv2d(3, 3, 1)
        conv2.weight.data = torch.eye(3)[..., None, None]
        conv2.bias.data = torch.zeros(3)
        max2 = nn.MaxPool2d((2, 2), (2, 2))
        fc1 = nn.Linear(3, 1)
        fc1.weight.data = torch.ones((1, 3))
        fc1.bias.data = torch.zeros(1)
        max_positive = nn.Sequential(
            conv1, nn.ReLU(), max1, conv2, nn.ReLU(), max2, nn.Flatten(1), fc1
        ).eval()
        torch.onnx.export(
            max_positive,
            dummy_input,
            max_positive_path,
            input_names=["input"],
            dynamic_axes={"input": [0]},
        )

    max_positive_path = artifact_dir / "max_positive_nonstandard.onnx"
    if not max_positive_path.exists():
        dummy_input = torch.ones((1, 3, 4, 4))
        max1 = nn.MaxPool2d((2, 2), (2, 2))
        max2 = nn.MaxPool2d((2, 2), (2, 2))
        fc1 = nn.Linear(3, 1)
        fc1.weight.data = torch.ones((1, 3))
        fc1.bias.data = torch.zeros(1)
        max_positive = nn.Sequential(nn.ReLU(), max1, max2, nn.Flatten(1), fc1).eval()
        torch.onnx.export(
            max_positive,
            dummy_input,
            max_positive_path,
            input_names=["input"],
            dynamic_axes={"input": [0]},
        )

    # create models with specific operations
    # TODO


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)

        self.conv1.weight.data = torch.from_numpy(np.random.randn(1, 1, 3, 3)).float()
        self.conv1.bias.data = torch.from_numpy(np.random.randn(1)).float()
        self.conv2.weight.data = torch.from_numpy(np.random.randn(8, 1, 3, 3)).float()
        self.conv2.bias.data = torch.from_numpy(np.random.randn(8)).float()
        self.downsample.weight.data = torch.from_numpy(
            np.random.randn(8, 1, 3, 3)
        ).float()
        self.downsample.bias.data = torch.from_numpy(np.random.randn(8)).float()
        self.running_mean = torch.from_numpy(np.random.randn(1)).float()
        self.running_var = torch.from_numpy(np.random.randn(1)).float()
        self.bn_weight = torch.from_numpy(np.random.randn(1)).float()
        self.bn_bias = torch.from_numpy(np.random.randn(1)).float()

    def forward(self, x):
        res = self.downsample(x)

        x = F.batch_norm(
            x, self.running_mean, self.running_var, self.bn_weight, self.bn_bias
        )
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = x + res
        return x


def build_random_weight_artifacts():
    # creates a set of tiny networks with random weights

    artifact_dir = Path(__file__).parent / "networks"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # random fully connected network
    fc1 = nn.Linear(5, 10)
    fc2 = nn.Linear(10, 10)
    fc3 = nn.Linear(10, 2)
    for i in range(3):
        np.random.seed(i)
        path = artifact_dir / f"random_fc_{i}.onnx"
        if not path.exists():
            fc1.weight.data = torch.from_numpy(np.random.randn(10, 5)).float()
            fc1.bias.data = torch.from_numpy(np.random.randn(10)).float()
            fc2.weight.data = torch.from_numpy(np.random.randn(10, 10)).float()
            fc2.bias.data = torch.from_numpy(np.random.randn(10)).float()
            fc3.weight.data = torch.from_numpy(np.random.randn(2, 10)).float()
            fc3.bias.data = torch.from_numpy(np.random.randn(2)).float()
            random = nn.Sequential(
                fc1, nn.ReLU(), fc2, nn.ReLU(), fc3, nn.Softmax(dim=1)
            ).eval()
            dummy_input = torch.ones((1, 5))
            torch.onnx.export(
                random,
                dummy_input,
                path,
                input_names=["input"],
                dynamic_axes={"input": [0]},
            )

    # random conv network
    conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
    conv2 = nn.Conv2d(2, 3, kernel_size=2, stride=1, padding=0)
    fc1 = nn.Linear(12, 2)
    for i in range(3):
        np.random.seed(i)
        path = artifact_dir / f"random_conv_{i}.onnx"
        if not path.exists():
            conv1.weight.data = torch.from_numpy(np.random.randn(2, 1, 3, 3)).float()
            conv1.bias.data = torch.from_numpy(np.random.randn(2)).float()
            conv2.weight.data = torch.from_numpy(np.random.randn(3, 2, 2, 2)).float()
            conv2.bias.data = torch.from_numpy(np.random.randn(3)).float()
            fc1.weight.data = torch.from_numpy(np.random.randn(2, 12)).float()
            fc1.bias.data = torch.from_numpy(np.random.randn(2)).float()
            random = nn.Sequential(
                conv1, nn.ReLU(), conv2, nn.ReLU(), Flatten(), fc1, nn.Softmax(dim=1)
            ).eval()
            dummy_input = torch.ones((1, 1, 3, 3))
            torch.onnx.export(
                random,
                dummy_input,
                path,
                input_names=["input"],
                dynamic_axes={"input": [0]},
            )

    # random residual network
    for i in range(1):
        np.random.seed(i)
        path = artifact_dir / f"random_residual_{i}.onnx"
        if not path.exists():
            res = Residual()
            fc1 = nn.Linear(8, 2)
            fc1.weight.data = torch.from_numpy(np.random.randn(2, 8)).float()
            fc1.bias.data = torch.from_numpy(np.random.randn(2)).float()
            random = nn.Sequential(
                res, nn.ReLU(), Flatten(), fc1, nn.Softmax(dim=1)
            ).eval()
            dummy_input = torch.ones((1, 1, 3, 3))
            torch.onnx.export(
                random,
                dummy_input,
                path,
                input_names=["input"],
                dynamic_axes={"input": [0]},
            )


def build():
    build_known_behavior_artifacts()
    build_random_weight_artifacts()


if __name__ == "__main__":
    build()
