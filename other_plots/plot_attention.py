import os
import pickle
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageOps


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)


# -----------------------------
# Global parameters for easy reuse
# -----------------------------
# Use one global root variable to switch source data quickly:
# RESULT_ROOT = os.path.join(PROJECT_ROOT, "generated_results", "SD2.1", "coco")
# RESULT_ROOT = os.path.join(PROJECT_ROOT, "generated_results", "SDXL", "coco")
# RESULT_ROOT = os.path.join(PROJECT_ROOT, "generated_results", "SDXL", "diffusiondb")
RESULT_ROOT = os.path.join(PROJECT_ROOT, "generated_results", "SD2.1", "coco")

# UUID of one sample image. Set to None to auto-pick the first sample in small_steps_attn.
# The script reads:
# - attention: generated_results/.../small_steps_attn/{UUID}_attn_matrix.pkl
# - original image: generated_results/.../org_images/{UUID}.(png|jpg|jpeg|webp)
TARGET_UUID = "183201"

# 1-based step number. Use None to export all steps from the attention pkl.
ATTN_STEP_NUMBER: int | None = None

# Overlay settings
OVERLAY_ALPHA = 0.45
COLORMAP_NAME = "turbo"


ORG_IMAGE_DIR = os.path.join(RESULT_ROOT, "org_images")
ATTN_DIR = os.path.join(RESULT_ROOT, "small_steps_attn")
OUTPUT_DIR = os.path.join(RESULT_ROOT, "temp_visualizations")


def load_pickle(path: str) -> Any:
	if not os.path.exists(path):
		raise FileNotFoundError(f"File not found: {path}")
	with open(path, "rb") as f:
		return pickle.load(f)


def resolve_target_uuid(attn_dir: str, target_uuid: str | None) -> str:
	if target_uuid is not None and str(target_uuid).strip() != "":
		return str(target_uuid)

	if not os.path.isdir(attn_dir):
		raise FileNotFoundError(f"Attention directory not found: {attn_dir}")

	candidates = sorted(
		f for f in os.listdir(attn_dir) if f.endswith("_attn_matrix.pkl")
	)
	if len(candidates) == 0:
		raise FileNotFoundError(f"No *_attn_matrix.pkl found under {attn_dir}")

	return candidates[0].replace("_attn_matrix.pkl", "")


def find_org_image_path(image_dir: str, uuid: str) -> str:
	exts = [".png", ".jpg", ".jpeg", ".webp"]
	for ext in exts:
		candidate = os.path.join(image_dir, f"{uuid}{ext}")
		if os.path.exists(candidate):
			return candidate
	raise FileNotFoundError(
		f"No original image found for uuid={uuid} under {image_dir}. "
		f"Tried extensions: {', '.join(exts)}"
	)


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
	arr = np.asarray(arr, dtype=np.float32)
	v_min = float(arr.min())
	v_max = float(arr.max())
	if v_max - v_min > 1e-12:
		return (arr - v_min) / (v_max - v_min)
	return np.zeros_like(arr, dtype=np.float32)


def attn_to_heatmap_rgb(attn_2d: np.ndarray, out_h: int, out_w: int, cmap_name: str) -> np.ndarray:
	if attn_2d.ndim != 2:
		raise ValueError(f"Each attention matrix must be 2D, got shape={attn_2d.shape}")

	attn_norm = normalize_to_01(attn_2d)
	attn_gray = (attn_norm * 255.0).round().astype(np.uint8)
	attn_img = Image.fromarray(attn_gray, mode="L")
	if attn_img.size != (out_w, out_h):
		attn_img = attn_img.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)

	attn_resized = np.asarray(attn_img, dtype=np.float32) / 255.0
	cmap = colormaps[cmap_name]
	heat_rgba = cmap(attn_resized)
	heat_rgb = (heat_rgba[..., :3] * 255.0).round().astype(np.uint8)
	return heat_rgb


def blend_overlay(org_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
	if org_rgb.shape != heat_rgb.shape:
		raise ValueError(f"Shape mismatch for blend: org={org_rgb.shape}, heat={heat_rgb.shape}")
	alpha = float(np.clip(alpha, 0.0, 1.0))
	out = (1.0 - alpha) * org_rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
	return np.clip(np.round(out), 0, 255).astype(np.uint8)


def save_triplet(org_img: Image.Image, heat_rgb: np.ndarray, overlay_rgb: np.ndarray, save_path: str) -> None:
	heat_img = Image.fromarray(heat_rgb, mode="RGB")
	overlay_img = Image.fromarray(overlay_rgb, mode="RGB")

	w, h = org_img.size
	gap = 12
	canvas = Image.new("RGB", (w * 3 + gap * 4, h + gap * 2), color=(245, 245, 245))
	canvas.paste(ImageOps.contain(org_img.convert("RGB"), (w, h)), (gap, gap))
	canvas.paste(ImageOps.contain(heat_img, (w, h)), (gap * 2 + w, gap))
	canvas.paste(ImageOps.contain(overlay_img, (w, h)), (gap * 3 + w * 2, gap))
	canvas.save(save_path)


def main() -> None:
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	target_uuid = resolve_target_uuid(ATTN_DIR, TARGET_UUID)
	attn_path = os.path.join(ATTN_DIR, f"{target_uuid}_attn_matrix.pkl")
	org_image_path = find_org_image_path(ORG_IMAGE_DIR, target_uuid)

	attn_steps = load_pickle(attn_path)
	if not isinstance(attn_steps, (list, tuple)) or len(attn_steps) == 0:
		raise ValueError(f"Attention pkl must be a non-empty list/tuple, got {type(attn_steps)} from {attn_path}")

	org_img = Image.open(org_image_path).convert("RGB")
	org_rgb = np.asarray(org_img, dtype=np.uint8)
	out_h, out_w = org_rgb.shape[0], org_rgb.shape[1]

	if ATTN_STEP_NUMBER is None:
		step_indices = list(range(len(attn_steps)))
	else:
		if ATTN_STEP_NUMBER <= 0:
			raise ValueError(f"ATTN_STEP_NUMBER must be >= 1 or None, got {ATTN_STEP_NUMBER}")
		step_idx = ATTN_STEP_NUMBER - 1
		if step_idx >= len(attn_steps):
			raise IndexError(f"Requested step {ATTN_STEP_NUMBER}, but only {len(attn_steps)} steps exist in {attn_path}")
		step_indices = [step_idx]

	saved_files: list[str] = []
	for step_idx in step_indices:
		attn_2d = np.asarray(attn_steps[step_idx], dtype=np.float32)
		heat_rgb = attn_to_heatmap_rgb(attn_2d, out_h=out_h, out_w=out_w, cmap_name=COLORMAP_NAME)
		overlay_rgb = blend_overlay(org_rgb, heat_rgb, alpha=OVERLAY_ALPHA)

		step_num = step_idx + 1
		heat_path = os.path.join(OUTPUT_DIR, f"{target_uuid}_step{step_num:03d}_attn_heatmap.png")
		overlay_path = os.path.join(OUTPUT_DIR, f"{target_uuid}_step{step_num:03d}_attn_overlay.png")
		triplet_path = os.path.join(OUTPUT_DIR, f"{target_uuid}_step{step_num:03d}_attn_triplet.png")

		Image.fromarray(heat_rgb, mode="RGB").save(heat_path)
		Image.fromarray(overlay_rgb, mode="RGB").save(overlay_path)
		save_triplet(org_img, heat_rgb, overlay_rgb, triplet_path)

		saved_files.extend([heat_path, overlay_path, triplet_path])

	print("Saved attention visualizations:")
	for path in saved_files:
		print(path)


if __name__ == "__main__":
	main()
