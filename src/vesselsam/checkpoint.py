"""Portable inference loading without local configuration or training imports."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

from .models import build_model

MODEL_KEYS = frozenset({"backbone", "adapter", "rank", "alpha", "dilations", "branch_channels"})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, cuda, or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or install CUDA-enabled PyTorch")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index does not exist: {device.index}")
    return device


def restore_model(checkpoint: str | Path, base_checkpoint: str | Path, device: torch.device):
    """Read architecture and weights only; never use paths saved in a checkpoint.

    Both the existing compact adaptation format and a metadata-free inference
    format are accepted. The exact base file is verified before model creation.
    """
    path = Path(checkpoint).expanduser()
    base = Path(base_checkpoint).expanduser()
    for label, candidate in (("VesselSAM", path), ("Base", base)):
        if not candidate.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {candidate}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, Mapping):
        raise ValueError("Expected a VesselSAM checkpoint dictionary")
    if saved.get("format") == "vesselsam-inference-v1":
        config = saved.get("model_config")
    elif saved.get("format") == "vesselsam-adaptation-v1":
        legacy_config = saved.get("config")
        config = legacy_config.get("model") if isinstance(legacy_config, Mapping) else None
    else:
        raise ValueError("Expected a VesselSAM adaptation/inference checkpoint, not base SAM weights")
    if not isinstance(config, Mapping) or not isinstance(saved.get("model"), Mapping):
        raise ValueError("Checkpoint is missing its model configuration or adaptation state")
    required = {"backbone", "adapter", "rank", "alpha", "dilations"}
    if not required.issubset(config):
        raise ValueError(f"Missing model configuration: {sorted(required - set(config))}")
    # An allowlist drops saved file paths, initialization labels, and all runtime settings.
    model_config = {key: config[key] for key in MODEL_KEYS if key in config}
    expected_hash = saved.get("base_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Checkpoint is missing a valid base_sha256 fingerprint")
    if file_sha256(base) != expected_hash:
        raise ValueError("Base checkpoint SHA256 mismatch; provide the matching original base weights")
    model = build_model(model_config, base)
    model.load_adaptation(saved["model"])
    return model.to(device).eval()
