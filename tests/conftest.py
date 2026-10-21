import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_sam_factory(monkeypatch):
    from segment_anything import sam_model_registry
    from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

    def make_sam(checkpoint=None):
        assert checkpoint is None
        return Sam(
            image_encoder=ImageEncoderViT(
                img_size=64, patch_size=16, embed_dim=32, depth=2, num_heads=4,
                out_chans=16, window_size=2, global_attn_indexes=(1,), use_rel_pos=True,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=16, image_embedding_size=(4, 4), input_image_size=(64, 64), mask_in_chans=4,
            ),
            mask_decoder=MaskDecoder(
                transformer_dim=16,
                transformer=TwoWayTransformer(depth=1, embedding_dim=16, num_heads=4, mlp_dim=32),
                iou_head_hidden_dim=16,
            ),
            pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
        )

    monkeypatch.setitem(sam_model_registry, "vit_b", make_sam)
    return make_sam


@pytest.fixture
def checkpoint_files(tmp_path, tiny_sam_factory):
    from vesselsam.checkpoint import file_sha256
    from vesselsam.models import VesselSAM

    torch.manual_seed(11)
    sam = tiny_sam_factory()
    base = tmp_path / "base.pth"
    torch.save(sam.state_dict(), base)
    model = VesselSAM(sam, rank=2, alpha=4, branch_channels=3)
    # Nonzero adapters exercise the actual adaptation rather than just base SAM.
    for name, parameter in model.named_parameters():
        if name.endswith(".up.weight"):
            torch.nn.init.normal_(parameter, std=0.1)
    for name, buffer in model.named_buffers():
        if name.endswith("running_mean"):
            buffer.fill_(0.2)
        elif name.endswith("running_var"):
            buffer.fill_(1.5)
    state = {
        key: value.clone() for key, value in model.state_dict().items()
        if key.startswith("sam.mask_decoder.") or ".q_adapter." in key or ".v_adapter." in key
    }
    payload = {
        "format": "vesselsam-inference-v1",
        "model_config": {
            "backbone": "vit_b", "adapter": "atrous", "rank": 2,
            "alpha": 4.0, "dilations": [1, 6, 12, 18], "branch_channels": 3,
        },
        "model": state, "base_sha256": file_sha256(base),
    }
    adapted = tmp_path / "vesselsam.pt"
    torch.save(payload, adapted)
    return base, adapted, model, payload
