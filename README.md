# VesselSAM

**Parameter-efficient adaptation of SAM to aortic vessel segmentation with AtrousLoRA.**

VesselSAM adapts the Segment Anything Model (SAM) to aortic vessel segmentation in CT angiography. AtrousLoRA combines low-rank adaptation with atrous attention to capture multi-scale vessel features. This repository provides inference with a MedSAM-initialized SAM ViT-B model, a CT slice, and a user-supplied bounding box.

<p align="center">
  <img src="assets/atrouslora.svg" alt="Conceptual comparison of LoRA and AtrousLoRA" width="900">
</p>
<p align="center"><em>LoRA and AtrousLoRA.</em></p>

<details>
<summary>Atrous attention module</summary>
<p align="center">
  <img src="assets/atrous-attention.svg" alt="Atrous attention with multi-scale dilated convolutions and global pooling" width="1000">
</p>
</details>

## Installation

Use Python 3.10 or later and Git. Install the appropriate [PyTorch](https://pytorch.org/get-started/locally/) and torchvision builds for your CPU/CUDA environment, then run from this repository:

```bash
git clone https://github.com/vikkki/VesselSAM.git
cd VesselSAM
python -m pip install -e ".[sam]"
```

## Model weights

**Hugging Face model release: coming soon.** No weights are bundled or downloaded automatically. The download link will be added here when the model is released.

Download the VesselSAM adaptation checkpoint when available and the **matching MedSAM ViT-B base checkpoint** from [MedSAM](https://github.com/bowang-lab/MedSAM). Save them anywhere and pass both paths explicitly. The base file's SHA256 must match the adaptation checkpoint. See [checkpoint details](checkpoints/README.md).

## Run inference

```bash
python -m vesselsam predict \
  --checkpoint /path/to/vesselsam.pt \
  --base-checkpoint /path/to/medsam_vit_b.pth \
  --image /path/to/slice.png \
  --box 120 80 340 410 \
  --output-dir outputs/prediction
```

- `--box`: `xmin ymin xmax ymax` in the **original image's pixel coordinates**, with x along columns and y along rows. Replace the example with your own box; coordinates must lie between `0` and `width-1` / `height-1`.
- Input: an 8-bit PNG/JPEG already windowed to **width 400 HU / level 40 HU**, or a 2D NPY/NPZ array in `[0,1]` (NPZ key: `image`). RGB images are converted to grayscale.
- For an array already expressed in **Hounsfield units**, use `--image /path/to/slice_hu.npy --raw-hu` (also supports NPZ). This clips to `[-160,240]` HU and maps to `[0,1]`; stored-value offsets must be corrected beforehand.
- `--device auto` selects CUDA when available and otherwise CPU. Use `--device cpu` or `--device cuda:0` to choose explicitly.

Outputs are `mask.png` (background `0`, foreground `255`), `probability.npy` (float32), and `prediction.json`, all in the output directory. The mask and probabilities retain the input slice's height and width. `--threshold` defaults to `0.5`.

Inference uses one external box per slice. Prepare slices at the expected 1 mm in-plane spacing before inference; the CLI resizes to the model grid but does not resample physical spacing, read 3D volumes, or locate vessels automatically.

## Citation

Zhao, X., Zhuang, P., Zhang, Z. *et al.* **VesselSAM: a parameter-efficient adaptation of SAM to aortic vessel segmentation with AtrousLoRA.** *BMC Medical Imaging* (2026). [Article / DOI](https://doi.org/10.1186/s12880-026-02911-1).

## Acknowledgments and license

Built on [SAM](https://github.com/facebookresearch/segment-anything) and [MedSAM](https://github.com/bowang-lab/MedSAM). Source code is covered by the included [MIT license](LICENSE); upstream projects and model weights retain their respective licenses.
