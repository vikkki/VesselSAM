"""Single-slice inference from a CT image and a user-supplied bounding box."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .checkpoint import restore_model, select_device


def load_image(path: str | Path, raw_hu: bool = False) -> np.ndarray:
    """Read a 2D input; raw CT uses a fixed 400 HU width / 40 HU level."""
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if raw_hu and suffix not in {".npy", ".npz"}:
        raise ValueError("--raw-hu requires a 2D NPY or NPZ array; PNG/JPEG do not preserve HU")
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "image" not in archive:
                raise ValueError("NPZ input must contain an 'image' array")
            image = np.asarray(archive["image"], dtype=np.float32)
    elif suffix == ".npy":
        image = np.load(path, allow_pickle=False).astype(np.float32)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as source:
            if source.mode not in {"L", "RGB"}:
                raise ValueError("PNG/JPEG input must be an 8-bit grayscale or RGB image")
            image = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
    else:
        raise ValueError("Supported input formats: .png, .jpg, .jpeg, .npy, .npz")
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("Input must be a nonempty, finite 2D image")
    if raw_hu:
        image = (np.clip(image, -160.0, 240.0) + 160.0) / 400.0
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Expected values in [0,1]; use --raw-hu for CT arrays already expressed in HU")
    return np.ascontiguousarray(image, dtype=np.float32)


def validate_box(box, height: int, width: int) -> np.ndarray:
    coordinates = np.asarray(box, dtype=np.float32)
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        raise ValueError("Box must contain four finite xyxy coordinates")
    xmin, ymin, xmax, ymax = coordinates
    if not (0 <= xmin < xmax <= width - 1 and 0 <= ymin < ymax <= height - 1):
        raise ValueError(
            f"Box must satisfy 0 <= xmin < xmax <= {width - 1} and "
            f"0 <= ymin < ymax <= {height - 1} in original image pixels"
        )
    return coordinates


@torch.inference_mode()
def predict(
    checkpoint: str | Path, base_checkpoint: str | Path,
    image_path: str | Path, box: list[float], output_dir: str | Path,
    device_name: str = "auto", raw_hu: bool = False, threshold: float = 0.5,
) -> Path:
    image = load_image(image_path, raw_hu)
    height, width = image.shape
    coordinates = validate_box(box, height, width)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must be strictly between zero and one")
    device = select_device(device_name)
    model = restore_model(checkpoint, base_checkpoint, device)
    size = int(model.sam.image_encoder.img_size)
    tensor = torch.from_numpy(image.copy())[None, None]
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    tensor = tensor.repeat(1, 3, 1, 1).to(device)
    scaled_box = coordinates * np.asarray([size / width, size / height] * 2, dtype=np.float32)
    boxes = torch.from_numpy(scaled_box.copy())[None].to(device)
    probability = model.predict_proba(tensor, boxes, output_size=(height, width))[0, 0].cpu().numpy()
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray((probability > threshold).astype(np.uint8) * 255).save(destination / "mask.png")
    np.save(destination / "probability.npy", probability.astype(np.float32))
    metadata = {
        "image": Path(image_path).name,
        "box_xyxy": coordinates.tolist(), "threshold": threshold,
        "shape_yx": [height, width], "device": str(device),
        "preprocessing": "HU window: width=400, level=40" if raw_hu else "normalized [0,1] input",
    }
    (destination / "prediction.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination / "mask.png"
