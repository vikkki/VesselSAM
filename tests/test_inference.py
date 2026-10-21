import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from vesselsam.checkpoint import restore_model
from vesselsam.cli import main, parser
from vesselsam.inference import load_image, predict, validate_box


def test_checkpoint_roundtrip_and_batch_pairing(checkpoint_files):
    base, adapted, reference, _ = checkpoint_files
    restored = restore_model(adapted, base, torch.device("cpu"))
    images = torch.rand(2, 3, 64, 64)
    boxes = torch.tensor([[3., 4., 40., 55.], [11., 9., 60., 45.]])
    actual = restored(images, boxes)
    torch.testing.assert_close(actual, reference(images, boxes), rtol=0, atol=0)
    separate = torch.cat([restored(images[i:i + 1], boxes[i:i + 1]) for i in range(2)])
    torch.testing.assert_close(actual, separate, rtol=1e-4, atol=1e-5)
    expected = F.interpolate(actual.sigmoid(), (23, 41), mode="bilinear", align_corners=False)
    torch.testing.assert_close(restored.predict_proba(images, boxes, (23, 41)), expected)
    assert not actual.requires_grad
    assert all(not p.requires_grad for p in restored.parameters())
    assert all(not module.training for module in restored.modules())


def test_legacy_checkpoint_ignores_saved_paths_and_runtime_settings(checkpoint_files, monkeypatch):
    import vesselsam.checkpoint as loading

    base, adapted, _, payload = checkpoint_files
    architecture = payload.pop("model_config")
    payload["format"] = "vesselsam-adaptation-v1"
    payload["config"] = {
        "model": {**architecture, "checkpoint": "unavailable.pth", "activation_checkpointing": True},
        "data": {"manifest": "unavailable.csv"},
    }
    payload["optimizer"] = {"unused": torch.tensor([1.0])}
    torch.save(payload, adapted)
    original = loading.build_model

    def inspect(config, path):
        assert config == architecture
        assert path == base
        return original(config, path)

    monkeypatch.setattr(loading, "build_model", inspect)
    loading.restore_model(adapted, base, torch.device("cpu"))


@pytest.mark.parametrize("corruption,match", [
    ("hash", "SHA256 mismatch"), ("missing_key", "missing keys"),
    ("extra_key", "unexpected keys"), ("shape", "tensor"),
    ("missing_config", "configuration"), ("format", "Expected a VesselSAM"),
])
def test_checkpoint_rejects_incompatible_weights(checkpoint_files, corruption, match):
    base, adapted, _, payload = checkpoint_files
    key = next(iter(payload["model"]))
    if corruption == "hash":
        payload["base_sha256"] = "0" * 64
    elif corruption == "missing_key":
        payload["model"].pop(key)
    elif corruption == "extra_key":
        payload["model"]["unexpected"] = torch.zeros(1)
    elif corruption == "shape":
        payload["model"][key] = torch.zeros(1)
    elif corruption == "missing_config":
        payload["model_config"].pop("alpha")
    else:
        payload["format"] = "unknown"
    torch.save(payload, adapted)
    with pytest.raises(ValueError, match=match):
        restore_model(adapted, base, torch.device("cpu"))


def test_raw_hu_and_normalized_formats(tmp_path):
    values = np.array([[-1024, -160, 0], [40, 240, 3071]], dtype=np.float32)
    expected = np.array([[0, 0, 0.4], [0.5, 1, 1]], dtype=np.float32)
    np.save(tmp_path / "hu.npy", values)
    np.savez(tmp_path / "hu.npz", image=values)
    for filename in ("hu.npy", "hu.npz"):
        np.testing.assert_allclose(load_image(tmp_path / filename, raw_hu=True), expected)
    np.save(tmp_path / "normalized.npy", expected)
    np.testing.assert_array_equal(load_image(tmp_path / "normalized.npy"), expected)
    pixels = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "slice.png")
    np.testing.assert_allclose(load_image(tmp_path / "slice.png"), pixels / 255.0)
    with pytest.raises(ValueError, match="raw-hu"):
        load_image(tmp_path / "slice.png", raw_hu=True)


@pytest.mark.parametrize("values", [
    np.empty((0, 4)), np.zeros((2, 3, 4)), np.array([[np.nan]]),
    np.array([[np.inf]]), np.array([[-1.0]]), np.array([[256.0]]),
])
def test_invalid_arrays_fail_before_model_loading(tmp_path, values):
    path = tmp_path / "invalid.npy"
    np.save(path, values)
    with pytest.raises(ValueError):
        predict("missing.pt", "missing.pth", path, [0, 0, 1, 1], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_npz_key_and_high_bit_depth_png_rejected(tmp_path):
    np.savez(tmp_path / "wrong.npz", mask=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="'image'"):
        load_image(tmp_path / "wrong.npz")
    Image.fromarray(np.ones((2, 2), dtype=np.uint16) * 1024).save(tmp_path / "16bit.png")
    with pytest.raises(ValueError, match="8-bit"):
        load_image(tmp_path / "16bit.png")


@pytest.mark.parametrize("box", [
    [-1, 0, 5, 5], [0, 0, 20, 5], [5, 0, 5, 5], [0, 0, 5, 10],
    [0, 0, np.nan, 5], [0, 0, 5],
])
def test_invalid_boxes(box):
    with pytest.raises(ValueError):
        validate_box(box, height=10, width=20)


@pytest.mark.parametrize("threshold", [0, 1, float("nan")])
def test_invalid_threshold(tmp_path, threshold):
    path = tmp_path / "slice.npy"
    np.save(path, np.zeros((10, 20), dtype=np.float32))
    with pytest.raises(ValueError, match="Threshold"):
        predict("missing.pt", "missing.pth", path, [0, 0, 19, 9], tmp_path / "out", threshold=threshold)


def test_cli_writes_original_resolution_predictions(tmp_path, checkpoint_files, capsys):
    base, adapted, reference, _ = checkpoint_files
    pixels = np.arange(23 * 41, dtype=np.uint8).reshape(23, 41)
    path = tmp_path / "slice.png"
    Image.fromarray(pixels).save(path)
    destination = tmp_path / "output"
    main([
        "predict", "--checkpoint", str(adapted), "--base-checkpoint", str(base),
        "--image", str(path), "--box", "2", "3", "39", "21",
        "--output-dir", str(destination), "--device", "cpu", "--threshold", "0.6",
    ])
    assert "Saved mask:" in capsys.readouterr().out
    # Independent coordinate/resize calculation catches axis swaps and mask resizing.
    tensor = torch.tensor(pixels.astype(np.float32) / 255.0)[None, None]
    tensor = F.interpolate(tensor, (64, 64), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    box = torch.tensor([[2 * 64 / 41, 3 * 64 / 23, 39 * 64 / 41, 21 * 64 / 23]])
    expected = reference.predict_proba(tensor, box, (23, 41))[0, 0].numpy()
    probability = np.load(destination / "probability.npy")
    assert probability.dtype == np.float32
    np.testing.assert_allclose(probability, expected, atol=1e-6)
    with Image.open(destination / "mask.png") as image:
        np.testing.assert_array_equal(np.asarray(image), (expected > 0.6).astype(np.uint8) * 255)
    metadata = json.loads((destination / "prediction.json").read_text())
    assert metadata["shape_yx"] == [23, 41]
    assert metadata["box_xyxy"] == [2, 3, 39, 21]
    assert metadata["image"] == "slice.png"


def test_cli_requires_explicit_paths_and_only_exposes_predict():
    assert "train" not in parser().format_help()
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["predict", "--checkpoint", "model.pt", "--image", "slice.png", "--box", "0", "0", "9", "9"])
    assert error.value.code == 2
