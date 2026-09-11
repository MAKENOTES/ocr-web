"""导出含预处理契约的文档专用 ONNX，并验证 CPU 数值一致性。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from detection_larger.models.east_model import ONNX_CONFIG_KEY, load_checkpoint


def export_model(checkpoint_path, output):
    import onnx
    import onnxruntime as ort

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"ONNX output already exists: {output}")
    model, checkpoint = load_checkpoint(checkpoint_path)
    height, width = checkpoint["config"]["input_size"]
    sample = torch.rand(1, 3, height, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, str(output), input_names=["image"], output_names=["score", "geometry"],
                      opset_version=17, dynamo=False, dynamic_axes={"image": {0: "batch"}, "score": {0: "batch"}, "geometry": {0: "batch"}})
    graph = onnx.load(str(output))
    onnx.helper.set_model_props(graph, {ONNX_CONFIG_KEY: json.dumps(checkpoint["config"])})
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(sample)
    actual = session.run(["score", "geometry"], {"image": sample.numpy()})
    for reference, observed in zip(expected, actual):
        np.testing.assert_allclose(observed, reference.numpy(), rtol=1e-4, atol=1e-5)
    return checkpoint["config"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="文档专用 .pth")
    parser.add_argument("--output", type=Path, default=Path("detection_larger/artifacts/best_detection.onnx"), help="新的 ONNX 路径")
    args = parser.parse_args()
    config = export_model(args.checkpoint, args.output)
    print(f"Export and numerical validation passed: {args.output}; {config}")


if __name__ == "__main__":
    main()
