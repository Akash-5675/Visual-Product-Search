"""ONNX export with a parity check against the PyTorch model."""
from pathlib import Path

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, out_path: Path, image_size: int = 224,
                opset: int = 17) -> Path:
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, image_size, image_size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["image"], output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    return out_path


def check_parity(model: torch.nn.Module, onnx_path: Path, n: int = 4,
                 image_size: int = 224, atol: float = 1e-4) -> float:
    """Max abs difference between torch and onnxruntime embeddings on random input."""
    import onnxruntime as ort
    model = model.eval().cpu()
    x = torch.randn(n, 3, image_size, image_size)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(["embedding"], {"image": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    assert diff < atol, f"onnx/torch mismatch: max abs diff {diff}"
    return diff
