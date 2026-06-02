#!/usr/bin/env python3
"""
Batch inference script for RADSeg: run zero-shot segmentation on images or directories.
Outputs segmentation masks (index or color), optional overlay, and optional probability maps.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference_utils import (
    parse_args as _parse_args,
    load_classes,
    get_image_paths,
    get_palette,
    mask_to_color,
    overlay_mask_on_image,
)


def parse_args():
    return _parse_args()


def resize_image_max_side(img: np.ndarray, max_side: int):
    """Resize image so the longer side is max_side; keep aspect ratio. Return (resized, orig_h, orig_w)."""
    h, w = img.shape[:2]
    orig_h, orig_w = h, w
    if max(h, w) <= max_side:
        return img, orig_h, orig_w
    scale = max_side / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    new_w = max(16, new_w)
    new_h = max(16, new_h)
    pil_img = Image.fromarray(img)
    resized = np.array(pil_img.resize((new_w, new_h), Image.BILINEAR))
    return resized, orig_h, orig_w


def image_to_tensor(img: np.ndarray, device):
    """Convert HWC uint8/float to 1xCxHxW float in [0,1]."""
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor


def run_inference(args):
    from radseg.radseg import RADSegEncoder

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.classes)
    if not classes:
        raise ValueError("No classes provided. Use --classes 'a,b,c' or path to a text file.")

    image_paths = get_image_paths(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    slide_crop = 0 if args.no_sliding else args.slide_crop
    slide_stride = args.slide_stride

    encoder = RADSegEncoder(
        device=device,
        model_version=args.model_version,
        lang_model=args.lang_model,
        scra_scaling=args.scra_scaling,
        scga_scaling=args.scga_scaling,
        slide_crop=slide_crop,
        slide_stride=slide_stride,
        predict=True,
        classes=classes,
        sam_refinement=args.sam_refine,
    )

    num_classes = len(classes)
    palette = get_palette(num_classes + 1)

    for path in image_paths:
        stem = path.stem
        img = np.array(Image.open(path).convert("RGB"))
        img_resized, orig_h, orig_w = resize_image_max_side(img, args.resolution)
        tensor = image_to_tensor(img_resized, device)
        with torch.inference_mode():
            seg_probs, seg_pred = encoder.encode_image_to_feat_map(
                tensor,
                orig_img_size=(orig_h, orig_w),
                return_preds=True,
                ignore_label=True,
            )
        seg_pred = seg_pred[0].cpu().numpy()
        seg_probs_np = seg_probs[0].cpu().numpy()

        if args.mask_format == "index":
            mask_save = seg_pred.astype(np.uint8)
            Image.fromarray(mask_save).save(os.path.join(args.output_dir, f"{stem}_mask.png"))
        else:
            color_mask = mask_to_color(seg_pred, palette)
            Image.fromarray(color_mask).save(os.path.join(args.output_dir, f"{stem}_mask.png"))

        if args.save_overlay:
            if args.mask_format != "color":
                color_mask = mask_to_color(seg_pred, palette)
            orig_img = np.array(Image.open(path).convert("RGB"))
            over = overlay_mask_on_image(orig_img, color_mask)
            Image.fromarray(over).save(os.path.join(args.output_dir, f"{stem}_overlay.png"))

        if args.save_probs:
            np.save(os.path.join(args.output_dir, f"{stem}_probs.npy"), seg_probs_np)

    print(f"Done. Processed {len(image_paths)} image(s). Outputs in {args.output_dir}")
    return 0


def main():
    args = parse_args()
    try:
        return run_inference(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
