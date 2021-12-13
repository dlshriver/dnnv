import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Any, Callable, Dict, List


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, 4)
        self.lin2 = nn.Linear(4, 4)
        self.lin3 = nn.Linear(4, output_size)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x


def train(
    model: nn.Module,
    labeller: Callable[[torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    random_x: Callable[[int], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    args: argparse.Namespace,
) -> nn.Module:
    model.train()
    iterations = args.iterations
    batch_size = args.batch_size
    early_stopping_error = args.early_stopping_error
    loss_history_window: int = args.loss_history_window
    loss_history: List[float] = []
    for i in range(iterations):
        optimizer.zero_grad()
        x = random_x(batch_size)
        y_true = labeller(x)
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        loss_val = loss.item()
        loss_history.append(loss_val)
        loss_history_len = len(loss_history)
        if loss_history_len > loss_history_window:
            loss_history.pop(0)
        if loss_history_len < loss_history_window:
            avg_loss = float("inf")
        else:
            avg_loss = sum(loss_history) / loss_history_window
        print(f"[ITERATION {i:9d}] loss={loss_val:.6f} avg_loss={avg_loss:.6f}")
        if avg_loss < early_stopping_error:
            print(f"[STOPPING EARLY] loss is less than {early_stopping_error}")
            return model
        loss.backward()
        optimizer.step()
    return model


def get_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), args.learning_rate)


def main(args: argparse.Namespace) -> int:
    specs: Dict[str, Any] = {}
    with open(args.env_spec) as env_spec_file:
        exec(env_spec_file.read(), specs)

    state_generator: Callable[[int], torch.Tensor] = specs[args.state_generator]
    transition_fn: Callable[[torch.Tensor], torch.Tensor] = specs[args.transition_fn]

    x = state_generator(1)
    y = transition_fn(x)
    transition_model = Model(x.flatten(1).size(1), y.flatten(1).size(1))

    optimizer = get_optimizer(transition_model, args)
    train(
        transition_model,
        transition_fn,
        optimizer,
        state_generator,
        F.mse_loss,
        args,
    )
    transition_model.eval()
    torch.onnx.export(
        transition_model,
        x,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": [0]},
    )

    if args.state_to_inputs_fn is not None:
        print("error: state_to_inputs_fn is not currently supported")
        return 1

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct neural network approximations of a given environment specification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="path at which to save the trained model",
    )

    # environment specs
    env_spec_group = parser.add_argument_group("Environment specification")
    env_spec_group.add_argument(
        "env_spec",
        type=Path,
        help="python file containing the environment specifications",
    )
    env_spec_group.add_argument(
        "--Eo",
        "--transition_fn",
        type=str,
        default="transition_fn",
        dest="transition_fn",
        help="transition function name",
    )
    env_spec_group.add_argument(
        "--Ei",
        "--state_to_inputs_fn",
        type=str,
        default=None,
        dest="state_to_inputs_fn",
        help="environment state to controller inputs function name",
    )
    env_spec_group.add_argument(
        "--sg",
        "--state-generator",
        type=str,
        default="state_generator",
        dest="state_generator",
        help="a function to generate states and control signals",
    )

    # training parameters
    training_group = parser.add_argument_group("Training parameters")
    training_group.add_argument(
        "-N",
        "--iterations",
        type=int,
        default=1_000_000,
        help="number of training iterations",
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="training batch size",
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="optimizer learning rate",
    )
    training_group.add_argument(
        "--loss-history-window",
        type=int,
        default=10,
        help="number of iterations of loss history to keep",
    )
    training_group.add_argument(
        "--early-stopping-error",
        type=float,
        default=float("-inf"),
        help="error at which to stop training early",
    )

    return parser.parse_args()


if __name__ == "__main__":
    exit(main(parse_args()))
