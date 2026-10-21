"""The public command line exposes inference only."""

from __future__ import annotations

import argparse


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="VesselSAM: box-prompted aortic vessel segmentation")
    commands = root.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Segment one CT slice using an external bounding box")
    predict.add_argument("--checkpoint", required=True, help="Path to downloaded VesselSAM adaptation weights")
    predict.add_argument("--base-checkpoint", required=True, help="Path to matching SAM/MedSAM ViT-B base weights")
    predict.add_argument("--image", required=True, help="PNG/JPEG or 2D NPY/NPZ (key: image)")
    predict.add_argument("--box", type=float, nargs=4, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    predict.add_argument("--output-dir", default="outputs/prediction")
    predict.add_argument("--device", default="auto", help="auto (CUDA if available, otherwise CPU), cpu, cuda, cuda:N")
    predict.add_argument("--raw-hu", action="store_true", help="Apply the 400/40 CT window to NPY/NPZ HU values")
    predict.add_argument("--threshold", type=float, default=0.5)
    return root


def main(argv=None) -> None:
    root = parser()
    args = root.parse_args(argv)
    from .inference import predict

    try:
        result = predict(
            checkpoint=args.checkpoint, base_checkpoint=args.base_checkpoint,
            image_path=args.image, box=args.box, output_dir=args.output_dir,
            device_name=args.device, raw_hu=args.raw_hu, threshold=args.threshold,
        )
    except (OSError, ValueError) as error:
        root.exit(2, f"vesselsam: {error}\n")
    print(f"Saved mask: {result}")
