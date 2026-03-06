"""Shared utilities for run_inference.py (no radseg/torch dependency for testing)."""

import argparse
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="RADSeg batch inference: segment images with custom class labels."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single image or directory of images.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="Comma-separated class names (e.g. 'car,person,road') or path to a text file (one class per line or comma-separated).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./radseg_out",
        help="Output directory for masks and overlays (default: ./radseg_out).",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="c-radio_v3-b",
        help="RADIO model version (default: c-radio_v3-b).",
    )
    parser.add_argument(
        "--lang_model",
        type=str,
        default="siglip2",
        help="Language model (default: siglip2).",
    )
    parser.add_argument(
        "--scra_scaling",
        type=float,
        default=10.0,
        help="SCRA scaling factor (default: 10.0).",
    )
    parser.add_argument(
        "--scga_scaling",
        type=float,
        default=10.0,
        help="SCGA scaling factor (default: 10.0).",
    )
    parser.add_argument(
        "--sam_refine",
        action="store_true",
        help="Enable SAM mask refinement (RADSeg+).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Max side length for resizing before inference (default: 512). Use larger for quality, smaller for speed.",
    )
    parser.add_argument(
        "--slide_crop",
        type=int,
        default=336,
        help="Sliding window crop size (default: 336). Set to 0 with --no_sliding to disable.",
    )
    parser.add_argument(
        "--slide_stride",
        type=int,
        default=224,
        help="Sliding window stride (default: 224).",
    )
    parser.add_argument(
        "--no_sliding",
        action="store_true",
        help="Disable sliding window (single-crop inference).",
    )
    parser.add_argument(
        "--mask-format",
        type=str,
        choices=("index", "color"),
        default="color",
        help="Mask output: 'index' (single-channel PNG, pixel=class index) or 'color' (RGB with palette) (default: color).",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save overlay of colored mask on original image.",
    )
    parser.add_argument(
        "--save-probs",
        action="store_true",
        help="Save per-class probability maps as .npy (one file per image: stem_probs.npy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available else cpu).",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def load_classes(classes_arg: str):
    """Load class names from comma-separated string or from a text file."""
    path = Path(classes_arg)
    if path.is_file():
        with open(path, "r") as f:
            text = f.read()
        names = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            for part in line.split(","):
                part = part.strip()
                if part:
                    names.append(part)
        return names
    names = [s.strip() for s in classes_arg.split(",") if s.strip()]
    return names


def get_image_paths(input_path: str):
    """Return list of image paths from a file or directory."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if p.is_file():
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {p.suffix}. Use one of {IMAGE_EXTENSIONS}"
            )
        return [p]
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(p.glob(f"*{ext}"))
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images found in directory: {input_path}")
    return paths


def get_palette(num_classes_plus_background: int):
    """Return RGB palette for class indices (0 = background/ignore). Deterministic distinct colors."""
    colors = [(0, 0, 0)]
    for i in range(1, num_classes_plus_background):
        hue = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colors, dtype=np.uint8)


def mask_to_color(mask: np.ndarray, palette: np.ndarray):
    """mask: HxW int (0..num_classes). palette: (num_classes+1, 3). Returns HxWx3 uint8."""
    h, w = mask.shape
    flat = np.clip(mask, 0, len(palette) - 1).ravel()
    rgb = palette[flat].reshape(h, w, 3)
    return rgb


def overlay_mask_on_image(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.5):
    """Blend color_mask on image. image and color_mask HxWx3 uint8."""
    if image.shape[:2] != color_mask.shape[:2]:
        color_mask = np.array(
            Image.fromarray(color_mask).resize(
                (image.shape[1], image.shape[0]),
                Image.NEAREST,
            )
        )
    blend = (image.astype(np.float32) * (1 - alpha) + color_mask.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return blend
