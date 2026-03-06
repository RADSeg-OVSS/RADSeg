"""Unit and smoke tests for scripts/run_inference.py (uses inference_utils only, no radseg)."""

import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from scripts.inference_utils import (
    load_classes,
    get_image_paths,
    get_palette,
    mask_to_color,
    overlay_mask_on_image,
    parse_args,
    IMAGE_EXTENSIONS,
)


class TestLoadClasses(unittest.TestCase):
    def test_comma_separated(self):
        self.assertEqual(load_classes("car,person,road"), ["car", "person", "road"])
        self.assertEqual(load_classes(" sky , grass "), ["sky", "grass"])

    def test_from_file_one_per_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("car\nperson\nroad\n")
            path = f.name
        try:
            self.assertEqual(load_classes(path), ["car", "person", "road"])
        finally:
            os.unlink(path)

    def test_from_file_comma_separated_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("car, person, road")
            path = f.name
        try:
            self.assertEqual(load_classes(path), ["car", "person", "road"])
        finally:
            os.unlink(path)


class TestGetImagePaths(unittest.TestCase):
    def test_single_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            paths = get_image_paths(path)
            self.assertEqual(len(paths), 1)
            self.assertEqual(Path(paths[0]).name, Path(path).name)
        finally:
            os.unlink(path)

    def test_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            get_image_paths("/nonexistent/path/12345")

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError):
                get_image_paths(path)
        finally:
            os.unlink(path)

    def test_directory(self):
        d = tempfile.mkdtemp()
        try:
            Path(d).joinpath("a.jpg").touch()
            Path(d).joinpath("b.png").touch()
            Path(d).joinpath("readme.txt").touch()
            paths = get_image_paths(d)
            self.assertEqual(len(paths), 2)
            names = {Path(p).name for p in paths}
            self.assertEqual(names, {"a.jpg", "b.png"})
        finally:
            for f in Path(d).iterdir():
                f.unlink()
            os.rmdir(d)

    def test_empty_directory_raises(self):
        d = tempfile.mkdtemp()
        try:
            with self.assertRaises(ValueError):
                get_image_paths(d)
        finally:
            os.rmdir(d)


class TestPaletteAndMask(unittest.TestCase):
    def test_palette_shape(self):
        p = get_palette(4)
        self.assertEqual(p.shape, (4, 3))
        self.assertEqual(p.dtype, "uint8")

    def test_mask_to_color(self):
        import numpy as np
        palette = get_palette(3)
        mask = np.array([[0, 1], [2, 0]], dtype=np.int32)
        rgb = mask_to_color(mask, palette)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_overlay_mask_on_image(self):
        import numpy as np
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        color_mask = np.zeros((10, 10, 3), dtype=np.uint8)
        color_mask[2:5, 2:5] = 255
        out = overlay_mask_on_image(img, color_mask, alpha=0.5)
        self.assertEqual(out.shape, (10, 10, 3))
        self.assertEqual(out.dtype, np.uint8)


class TestParseArgs(unittest.TestCase):
    def test_missing_required_raises(self):
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_minimal_required(self):
        args = parse_args([
            "--input", "/some/img.jpg",
            "--classes", "a,b,c",
        ])
        self.assertEqual(args.input, "/some/img.jpg")
        self.assertEqual(args.classes, "a,b,c")
        self.assertEqual(args.output_dir, "./radseg_out")
        self.assertEqual(args.model_version, "c-radio_v3-b")
        self.assertFalse(args.sam_refine)
        self.assertEqual(args.mask_format, "color")

    def test_options(self):
        args = parse_args([
            "--input", "/x", "--classes", "a,b",
            "--output-dir", "/out",
            "--resolution", "768",
            "--no_sliding",
            "--mask-format", "index",
            "--save-overlay", "--save-probs", "--sam_refine",
        ])
        self.assertEqual(args.output_dir, "/out")
        self.assertEqual(args.resolution, 768)
        self.assertTrue(args.no_sliding)
        self.assertEqual(args.mask_format, "index")
        self.assertTrue(args.save_overlay)
        self.assertTrue(args.save_probs)
        self.assertTrue(args.sam_refine)
