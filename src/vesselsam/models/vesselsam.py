"""Inference wrapper with state keys compatible with VesselSAM adaptations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .adapters import QKVWithAdapters


class VesselSAM(nn.Module):
    """SAM ViT-B with independent AtrousLoRA paths in the Q/V projections.

    Images are square RGB tensors in [0, 1]. Box prompts use xyxy coordinates
    on the resized image. The MedSAM-style encoder input bypasses SAM's
    natural-image pixel_mean/pixel_std normalization.
    """

    def __init__(
        self,
        sam: nn.Module,
        rank: int = 4,
        alpha: float = 4.0,
        dilations: Sequence[int] = (1, 6, 12, 18),
        branch_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.sam = sam
        for block in self.sam.image_encoder.blocks:
            if isinstance(block.attn.qkv, QKVWithAdapters):
                raise TypeError("This SAM instance already contains Q/V adapters")
            block.attn.qkv = QKVWithAdapters(
                block.attn.qkv, rank=rank, alpha=alpha, atrous=True,
                dilations=dilations, branch_channels=branch_channels,
            )
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Return one foreground-mask logit map per image and box."""
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] < 1:
            raise ValueError("images must have nonempty shape [B,3,H,W]")
        if not images.is_floating_point():
            raise TypeError("images must be floating point and normalized to [0,1]")
        size = int(self.sam.image_encoder.img_size)
        if tuple(images.shape[-2:]) != (size, size):
            raise ValueError(f"image encoder requires {size}x{size} inputs")
        if boxes.ndim != 2 or tuple(boxes.shape) != (images.shape[0], 4):
            raise ValueError("boxes must have shape [B,4], one xyxy box per image")
        boxes = boxes.to(device=images.device, dtype=torch.float32)
        embeddings = self.sam.image_encoder(images)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        logits = []
        # SAM repeats one image for each prompt. Decode per image to avoid
        # accidentally pairing every image with every box in a batch.
        for index in range(images.shape[0]):
            mask, _ = self.sam.mask_decoder(
                image_embeddings=embeddings[index:index + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse[index:index + 1],
                dense_prompt_embeddings=dense[index:index + 1],
                multimask_output=False,
            )
            logits.append(mask)
        return torch.cat(logits, dim=0)

    @torch.inference_mode()
    def predict_proba(
        self, images: torch.Tensor, boxes: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Apply sigmoid before resizing probabilities to the input grid."""
        probability = self(images, boxes).sigmoid()
        if output_size is not None:
            probability = F.interpolate(
                probability, size=output_size, mode="bilinear", align_corners=False,
            )
        return probability

    def load_adaptation(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load all Q/V adapters, BN buffers, and the complete mask decoder."""
        current = self.state_dict()
        expected = {
            key for key in current
            if key.startswith("sam.mask_decoder.")
            or ".q_adapter." in key or ".v_adapter." in key
        }
        supplied = set(state)
        if supplied != expected:
            raise ValueError(
                "Incompatible adaptation state: "
                f"missing keys={sorted(expected - supplied)}; "
                f"unexpected keys={sorted(supplied - expected)}"
            )
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != current[key].shape:
                raise ValueError(f"Incompatible adaptation tensor at {key}")
        self.load_state_dict(state, strict=False)


def build_model(config: Mapping[str, Any], base_checkpoint: str | Path) -> VesselSAM:
    """Load an explicitly supplied base checkpoint before installing adapters."""
    if config.get("backbone", "vit_b") != "vit_b":
        raise ValueError("VesselSAM inference supports SAM ViT-B only")
    if config.get("adapter", "atrous") != "atrous":
        raise ValueError("This release supports AtrousLoRA checkpoints only")
    path = Path(base_checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {path}")
    try:
        from segment_anything import sam_model_registry
    except ImportError as error:
        raise ImportError("Install the official SAM dependency: pip install -e '.[sam]'") from error
    sam = sam_model_registry["vit_b"](checkpoint=None)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Base checkpoint must contain a SAM/MedSAM state dictionary")
    sam.load_state_dict(state, strict=True)
    return VesselSAM(
        sam, rank=config.get("rank", 4), alpha=config.get("alpha", 4.0),
        dilations=config.get("dilations", (1, 6, 12, 18)),
        branch_channels=config.get("branch_channels"),
    )
