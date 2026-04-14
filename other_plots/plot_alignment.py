import os
import pickle
import sys
from typing import Any

import math
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from config import Config
from pipeline import SD2Pipeline


# -----------------------------
# Global parameters for easy reuse
# -----------------------------
UUID_ORG = "33629"
ORG_STEP_NUMBER = 30
ORG_STEP_INDEX = ORG_STEP_NUMBER - 1

UUID_SMALL = UUID_ORG
SMALL_STEP_NUMBER = 4
SMALL_STEP_INDEX = SMALL_STEP_NUMBER - 1


RESULT_ROOT = os.path.join(PROJECT_ROOT, "generated_results", "SD2.1", "coco")

ORG_TRAJ_PKL = os.path.join(RESULT_ROOT, "org_images", f"{UUID_ORG}.pkl")
SMALL_TRAJ_PKL = os.path.join(RESULT_ROOT, "small_steps_images", f"{UUID_SMALL}.pkl")

OUTPUT_DIR = os.path.join(RESULT_ROOT, "temp_visualizations")


def load_pickle(path: str) -> Any:
	if not os.path.exists(path):
		raise FileNotFoundError(f"File not found: {path}")
	with open(path, "rb") as f:
		return pickle.load(f)


def validate_steps(traj_obj: Any, step_number: int, src_path: str) -> None:
	if not isinstance(traj_obj, (list, tuple)):
		raise TypeError(f"Expected list/tuple in {src_path}, got {type(traj_obj)}")
	if step_number <= 0:
		raise ValueError(f"step_number must be positive, got {step_number}")
	if len(traj_obj) < step_number:
		raise IndexError(f"{src_path} has only {len(traj_obj)} steps, cannot access 1..{step_number}")


def build_sd21_pipeline(device: torch.device) -> SD2Pipeline:
	model_path = Config.MODEL_CKPT_PATHS["SD2.1Pipeline"]
	dtype = torch.float16 if device.type == "cuda" else torch.float32
	pipe = SD2Pipeline.from_pretrained(model_path, torch_dtype=dtype)
	pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
	pipe = pipe.to(device)
	pipe.set_progress_bar_config(disable=True)
	return pipe


def decode_sd21_latent_to_pil(pipe: SD2Pipeline, latent_np: np.ndarray, device: torch.device) -> Image.Image:
	latents = torch.from_numpy(np.asarray(latent_np)).to(device=device, dtype=pipe.vae.dtype)
	if latents.ndim == 3:
		latents = latents.unsqueeze(0)
	if latents.ndim != 4:
		raise ValueError(f"Latent must be 4D [B,C,H,W], got shape={tuple(latents.shape)}")

	scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
	if scaling_factor is None:
		scaling_factor = 0.18215

	with torch.no_grad():
		image = pipe.vae.decode(latents / scaling_factor, return_dict=False)[0]
		image = (image / 2 + 0.5).clamp(0, 1)
		image = image[0].permute(1, 2, 0).float().cpu().numpy()

	image_uint8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
	return Image.fromarray(image_uint8)


def decode_steps(
	pipe: SD2Pipeline,
	traj_obj: list,
	step_number: int,
	device: torch.device,
) -> list[Image.Image]:
	images: list[Image.Image] = []
	for step_idx in range(step_number):
		latent = np.asarray(traj_obj[step_idx], dtype=np.float32)
		img = decode_sd21_latent_to_pil(pipe, latent, device)
		images.append(img)
	return images


def save_step_images(images: list[Image.Image], save_dir: str) -> None:
	os.makedirs(save_dir, exist_ok=True)
	for i, img in enumerate(images, start=1):
		img.save(os.path.join(save_dir, f"step_{i:03d}.png"))


def make_step_grid(images: list[Image.Image], cols: int = 5) -> Image.Image:
	if len(images) == 0:
		raise ValueError("Cannot build grid from empty image list")

	cols = max(1, min(cols, len(images)))
	rows = math.ceil(len(images) / cols)
	tile_w, tile_h = images[0].size
	pad = 8
	label_h = 24

	grid_w = cols * tile_w + (cols + 1) * pad
	grid_h = rows * (tile_h + label_h) + (rows + 1) * pad
	grid = Image.new("RGB", (grid_w, grid_h), color=(248, 248, 248))
	draw = ImageDraw.Draw(grid)

	for idx, img in enumerate(images, start=1):
		r = (idx - 1) // cols
		c = (idx - 1) % cols
		x = pad + c * (tile_w + pad)
		y = pad + r * (tile_h + label_h + pad)
		grid.paste(img, (x, y))
		draw.text((x + 6, y + tile_h + 4), f"step {idx}", fill=(30, 30, 30))

	return grid


def main() -> None:
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	org_traj = load_pickle(ORG_TRAJ_PKL)
	small_traj = load_pickle(SMALL_TRAJ_PKL)
	validate_steps(org_traj, ORG_STEP_NUMBER, ORG_TRAJ_PKL)
	validate_steps(small_traj, SMALL_STEP_NUMBER, SMALL_TRAJ_PKL)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	pipe = build_sd21_pipeline(device)

	org_images = decode_steps(pipe, org_traj, ORG_STEP_NUMBER, device)
	small_images = decode_steps(pipe, small_traj, SMALL_STEP_NUMBER, device)

	org_step_dir = os.path.join(OUTPUT_DIR, f"{UUID_ORG}_org_steps_1_to_{ORG_STEP_NUMBER}")
	small_step_dir = os.path.join(OUTPUT_DIR, f"{UUID_SMALL}_small_steps_1_to_{SMALL_STEP_NUMBER}")
	save_step_images(org_images, org_step_dir)
	save_step_images(small_images, small_step_dir)

	org_grid = make_step_grid(org_images, cols=5)
	small_grid = make_step_grid(small_images, cols=5)
	org_grid_path = os.path.join(OUTPUT_DIR, f"{UUID_ORG}_org_steps_1_to_{ORG_STEP_NUMBER}_decoded_grid.png")
	small_grid_path = os.path.join(OUTPUT_DIR, f"{UUID_SMALL}_small_steps_1_to_{SMALL_STEP_NUMBER}_decoded_grid.png")
	org_grid.save(org_grid_path)
	small_grid.save(small_grid_path)

	print("Saved decoded step images and summary grids:")
	print(f"ORG step images dir: {org_step_dir}")
	print(f"SMALL step images dir: {small_step_dir}")
	print(f"ORG grid: {org_grid_path}")
	print(f"SMALL grid: {small_grid_path}")


if __name__ == "__main__":
	main()

