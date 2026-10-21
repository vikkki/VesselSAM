# Model weights

Download the inference weights from **[Hugging Face](https://huggingface.co/hakuuux/VesselSAM_0.2)**. Model files are hosted there rather than in this GitHub repository.

For inference, download:

1. One VesselSAM AtrousLoRA checkpoint: [AVT6.pt](https://huggingface.co/hakuuux/VesselSAM_0.2/blob/main/AVT6.pt) for AVT or [TBAD10.pt](https://huggingface.co/hakuuux/VesselSAM_0.2/blob/main/TBAD10.pt) for TBAD.
2. Its matching MedSAM ViT-B base checkpoint (`medsam_vit_b.pth`), available through [MedSAM](https://github.com/bowang-lab/MedSAM).

Pass the files to `--checkpoint` and `--base-checkpoint`. They may be stored in any directory. The loader verifies the base file's SHA256, loads weights with `weights_only=True`, and validates every adaptation tensor. A base SAM/MedSAM checkpoint alone is insufficient.

## Supported checkpoint format

The inference release format is a PyTorch dictionary with these fields:

```python
{
    "format": "vesselsam-inference-v1",
    "model_config": {
        "backbone": "vit_b",
        "adapter": "atrous",
        "rank": 4,
        "alpha": 4.0,
        "dilations": [1, 6, 12, 18],
        "branch_channels": 4,
    },
    "model": ...,        # Q/V adapters, BN buffers, and the complete mask decoder
    "base_sha256": ...,  # SHA256 of the exact base checkpoint file
}
```

The architecture values above describe the default implementation; released weights must carry their own matching configuration. Adaptation state keys use `sam.image_encoder.blocks.*.attn.qkv.q_adapter.*`, `v_adapter.*`, and `sam.mask_decoder.*`.

Existing `vesselsam-adaptation-v1` compact checkpoints are also accepted, using `config.model` for architecture. Saved paths and runtime settings are ignored. For distribution on Hugging Face, use the inference format above and omit optimizer state, dataset information, and local settings.
