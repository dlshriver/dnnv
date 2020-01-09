def build_networks():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models

    from pathlib import Path

    artifact_dir = Path(__file__).parent / "networks"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # create several imagenet models
    imagenet_dummy_input = torch.ones((1, 3, 224, 224))

    resnet34_path = artifact_dir / "resnet34.onnx"
    if not resnet34_path.exists():
        resnet34 = models.resnet34(pretrained=True).eval()
        torch.onnx.export(resnet34, imagenet_dummy_input, resnet34_path)

    resnet50_path = artifact_dir / "resnet50.onnx"
    if not resnet50_path.exists():
        resnet50 = models.resnet50(pretrained=True).eval()
        torch.onnx.export(resnet50, imagenet_dummy_input, resnet50_path)

    vgg16_path = artifact_dir / "vgg16.onnx"
    if not vgg16_path.exists():
        vgg16 = models.vgg16(pretrained=True).eval()
        torch.onnx.export(vgg16, imagenet_dummy_input, vgg16_path)

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

    # create models with specific operations
    # TODO


if __name__ == "__main__":
    build_networks()
