import os
import pickle
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..config import Config
from ..pipeline import SD2Pipeline


# -----------------------------
# Global parameters for easy reuse
# -----------------------------
UUID_ORG = "381514"
UUID_SMALL = "21075"

# Human-friendly step number for 381514 (15th step -> index 14)
ORG_STEP_NUMBER = 20
ORG_STEP_INDEX = ORG_STEP_NUMBER - 1

# Explicit index for 535389 trajectory (3rd step -> index 2)
ORG_STEP_NUMBER = 5
SMALL_STEP_INDEX = ORG_STEP_NUMBER - 1


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.join(PROJECT_ROOT, "generated_results", "SD2.1", "coco")

ORG_TRAJ_PKL = os.path.join(RESULT_ROOT, "org_images", f"{UUID_ORG}.pkl")
SMALL_TRAJ_PKL = os.path.join(RESULT_ROOT, "small_steps_images", f"{UUID_SMALL}.pkl")
SMALL_ATTN_PKL = os.path.join(RESULT_ROOT, "small_steps_attn", f"{UUID_SMALL}_attn_matrix.pkl")

OUTPUT_DIR = os.path.join(RESULT_ROOT, "temp_visualizations")


def load_pickle(path: str) -> Any:
	if not os.path.exists(path):
		raise FileNotFoundError(f"File not found: {path}")
	with open(path, "rb") as f:
		return pickle.load(f)


def get_step_item(sequence_obj: Any, step_index: int, src_path: str) -> np.ndarray:
	if not isinstance(sequence_obj, (list, tuple)):
		raise TypeError(f"Expected list/tuple in {src_path}, got {type(sequence_obj)}")
	if step_index < 0 or step_index >= len(sequence_obj):
		raise IndexError(
			f"Step index {step_index} out of range for {src_path} (len={len(sequence_obj)})"
		)
	item = np.asarray(sequence_obj[step_index])
	return item


def decode_sd21_latent_to_pil(pipe: SD2Pipeline, latent_np: np.ndarray, device: torch.device) -> Image.Image:
	latents = torch.from_numpy(latent_np).to(device=device, dtype=pipe.vae.dtype)
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


def attention_to_image(attn_np: np.ndarray) -> Image.Image:
	if attn_np.ndim != 2:
		raise ValueError(f"Attention matrix must be 2D, got shape={tuple(attn_np.shape)}")
	attn = attn_np.astype(np.float32)
	v_min = float(attn.min())
	v_max = float(attn.max())
	if v_max - v_min > 1e-12:
		attn = (attn - v_min) / (v_max - v_min)
	else:
		attn = np.zeros_like(attn, dtype=np.float32)
	attn_uint8 = np.clip(np.round(attn * 255.0), 0, 255).astype(np.uint8)
	return Image.fromarray(attn_uint8, mode="L")


def build_sd21_pipeline(device: torch.device) -> SD2Pipeline:
	model_path = Config.MODEL_CKPT_PATHS["SD2.1Pipeline"]
	dtype = torch.float16 if device.type == "cuda" else torch.float32
	pipe = SD2Pipeline.from_pretrained(model_path, torch_dtype=dtype)
	pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
	pipe = pipe.to(device)
	pipe.set_progress_bar_config(disable=True)
	return pipe


def main() -> None:
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	pipe = build_sd21_pipeline(device)

	# 1) Decode 381514 trajectory at the 15th step (index 14)
	org_traj = load_pickle(ORG_TRAJ_PKL)
	org_step_latent = get_step_item(org_traj, ORG_STEP_INDEX, ORG_TRAJ_PKL)
	org_img = decode_sd21_latent_to_pil(pipe, org_step_latent, device)
	org_out = os.path.join(OUTPUT_DIR, f"{UUID_ORG}_step{ORG_STEP_NUMBER}_decoded.png")
	org_img.save(org_out)

	# 2) Decode 535389 small-step trajectory at index 2 (3rd step)
	small_traj = load_pickle(SMALL_TRAJ_PKL)
	small_step_latent = get_step_item(small_traj, SMALL_STEP_INDEX, SMALL_TRAJ_PKL)
	small_img = decode_sd21_latent_to_pil(pipe, small_step_latent, device)
	small_out = os.path.join(OUTPUT_DIR, f"{UUID_SMALL}_step{SMALL_STEP_INDEX + 1}_decoded.png")
	small_img.save(small_out)

	# 3) Visualize the 3rd-step attention matrix (index 2)
	attn_mats = load_pickle(SMALL_ATTN_PKL)
	attn_step = get_step_item(attn_mats, SMALL_STEP_INDEX, SMALL_ATTN_PKL)
	attn_img = attention_to_image(attn_step)
	attn_out = os.path.join(OUTPUT_DIR, f"{UUID_SMALL}_step{SMALL_STEP_INDEX + 1}_attn.png")
	attn_img.save(attn_out)

	print("Saved 3 images:")
	print(f"1) {org_out}")
	print(f"2) {small_out}")
	print(f"3) {attn_out}")


if __name__ == "__main__":
	main()
