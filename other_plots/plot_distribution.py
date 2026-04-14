import os
import pickle
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Set default font (prefer system-available fonts first) and increase base font sizes
# Use DejaVu Sans / Liberation Sans (commonly available), then fall back to Calibri/Arial if present.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Calibri", "Arial"]
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 20
# plt.rcParams["xtick.labelsize"] = 16
# plt.rcParams["ytick.labelsize"] = 16

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# -----------------------------
# Global parameters for easy reuse
# -----------------------------
PIPELINE_RESULT_DIR = "SDXL"
DATASET_NAME = "coco"

# UUID_ORG = "421876"
UUID_ORG = "558900"
ORG_STEP_NUMBER = 15
ORG_STEP_INDEX = ORG_STEP_NUMBER - 1

UUID_SMALL = "558900"
# SMALL_STEP_NUMBER = 15
SMALL_STEP_NUMBER = 3
SMALL_STEP_INDEX = SMALL_STEP_NUMBER - 1


RESULT_ROOT = os.path.join(PROJECT_ROOT, "generated_results", PIPELINE_RESULT_DIR, DATASET_NAME)

ORG_TRAJ_PKL = os.path.join(RESULT_ROOT, "org_images", f"{UUID_ORG}.pkl")
SMALL_TRAJ_PKL = os.path.join(RESULT_ROOT, "small_steps_images", f"{UUID_SMALL}.pkl")
# SMALL_TRAJ_PKL = os.path.join(RESULT_ROOT, "org_images", f"{UUID_SMALL}.pkl")

OUTPUT_DIR = os.path.join(RESULT_ROOT, "temp_visualizations")

MAX_POINTS_PER_LATENT = 1200
SHOW_PROGRESS = True
VISUAL_SEPARATION_MODE = True

# Larger value => stronger center-to-center push for display only.
CENTER_PUSH_MULTIPLIER = 2.8


def stage(msg: str) -> None:
    if SHOW_PROGRESS:
        print(f"[Stage] {msg}", flush=True)


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


def latent_to_points(latent_np: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent_np, dtype=np.float32)

    # Common case for SD latents: [B, C, H, W] -> [B*H*W, C]
    if latent.ndim == 4:
        b, c, h, w = latent.shape
        return latent.transpose(0, 2, 3, 1).reshape(b * h * w, c)

    # [C, H, W] -> [H*W, C]
    if latent.ndim == 3:
        c, h, w = latent.shape
        return latent.transpose(1, 2, 0).reshape(h * w, c)

    # Already point-like [N, D]
    if latent.ndim == 2:
        return latent

    # Fallback for unexpected shape
    return latent.reshape(-1, 1)


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"PCA input must be 2D, got shape={x.shape}")

    n_samples, _ = x.shape
    if n_samples == 0:
        raise ValueError("PCA input is empty")

    stage(f"running PCA on {n_samples} points")

    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)

    n_comp = min(2, s.shape[0])
    emb = u[:, :n_comp] * s[:n_comp]
    if n_comp < 2:
        emb = np.concatenate([emb, np.zeros((n_samples, 2 - n_comp), dtype=emb.dtype)], axis=1)

    if n_samples > 1:
        variances = (s ** 2) / (n_samples - 1)
    else:
        variances = np.zeros_like(s)
    total_var = float(np.sum(variances))

    explained_ratio = np.zeros(2, dtype=np.float64)
    if total_var > 1e-12:
        explained_ratio[:n_comp] = variances[:n_comp] / total_var

    return emb.astype(np.float32), explained_ratio


def run_pca(org_points: np.ndarray, small_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_points = np.concatenate([org_points, small_points], axis=0).astype(np.float64)

    # Normalize each feature dimension before PCA.
    mean = all_points.mean(axis=0, keepdims=True)
    std = all_points.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    all_points = (all_points - mean) / std

    emb, explained_ratio = pca_2d(all_points)
    org_emb = emb[: org_points.shape[0]]
    small_emb = emb[org_points.shape[0] :]
    return org_emb, small_emb, explained_ratio


def enhance_visual_separation(
    org_emb: np.ndarray,
    small_emb: np.ndarray,
    push_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    c_org = org_emb.mean(axis=0)
    c_small = small_emb.mean(axis=0)

    direction = c_org - c_small
    base_gap = float(np.linalg.norm(direction))
    if base_gap < 1e-9:
        direction = np.array([1.0, 0.0], dtype=np.float32)
        base_gap = 0.0
    else:
        direction = direction / (base_gap + 1e-12)

    # Estimate cluster spread to derive a target inter-center distance.
    spread_org = float(np.mean(np.linalg.norm(org_emb - c_org, axis=1)))
    spread_small = float(np.mean(np.linalg.norm(small_emb - c_small, axis=1)))
    spread = 0.5 * (spread_org + spread_small)

    target_gap = max(base_gap, push_multiplier * max(spread, 1e-6))
    delta = max(0.0, 0.5 * (target_gap - base_gap))

    org_out = org_emb + delta * direction[None, :]
    small_out = small_emb - delta * direction[None, :]
    new_gap = float(np.linalg.norm(org_out.mean(axis=0) - small_out.mean(axis=0)))
    return org_out, small_out, base_gap, new_gap


def plot_pca(
    org_emb: np.ndarray,
    small_emb: np.ndarray,
    explained_ratio: np.ndarray,
    overlay_path: str,
) -> None:
    pc1_ratio = explained_ratio[0] * 100.0
    pc2_ratio = explained_ratio[1] * 100.0
    # Single-overlay figure (no titles, no x/y labels)
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=160)
    ax1.scatter(org_emb[:, 0], org_emb[:, 1], s=14, alpha=0.6, c="#1f77b4", label=f"{UUID_ORG}: step {ORG_STEP_NUMBER}")
    ax1.scatter(small_emb[:, 0], small_emb[:, 1], s=14, alpha=0.6, c="#ff7f0e", label=f"{UUID_SMALL}: step {SMALL_STEP_NUMBER}")
    # remove title and axis labels for cleaner export
    ax1.set_title("")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.grid(alpha=0.2)
    # enlarge legend markers for better visibility
    ax1.legend(markerscale=3, scatterpoints=1, fontsize=plt.rcParams["legend.fontsize"])
    fig1.tight_layout()
    fig1.savefig(overlay_path)
    plt.close(fig1)


def main() -> None:
    stage("preparing paths")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stage(f"loading pkl files from {PIPELINE_RESULT_DIR}/{DATASET_NAME}")
    org_traj = load_pickle(ORG_TRAJ_PKL)
    small_traj = load_pickle(SMALL_TRAJ_PKL)
    validate_steps(org_traj, ORG_STEP_NUMBER, ORG_TRAJ_PKL)
    validate_steps(small_traj, SMALL_STEP_NUMBER, SMALL_TRAJ_PKL)

    stage("extracting selected latent steps")
    org_latent = np.asarray(org_traj[ORG_STEP_INDEX], dtype=np.float32)
    small_latent = np.asarray(small_traj[SMALL_STEP_INDEX], dtype=np.float32)

    stage("sampling points")
    org_points = sample_points(latent_to_points(org_latent), MAX_POINTS_PER_LATENT, seed=42)
    small_points = sample_points(latent_to_points(small_latent), MAX_POINTS_PER_LATENT, seed=43)

    stage("running dimensionality reduction")
    org_emb, small_emb, explained_ratio = run_pca(org_points, small_points)

    base_gap, new_gap = 0.0, 0.0
    if VISUAL_SEPARATION_MODE:
        stage("enhancing visual separation")
        org_emb, small_emb, base_gap, new_gap = enhance_visual_separation(
            org_emb,
            small_emb,
            push_multiplier=CENTER_PUSH_MULTIPLIER,
        )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        f"{UUID_ORG}_org_step{ORG_STEP_NUMBER}_vs_{UUID_SMALL}_small_step{SMALL_STEP_NUMBER}_pca_overlay.svg",
    )

    stage("saving plot")
    plot_pca(org_emb, small_emb, explained_ratio, overlay_path)

    print("Saved PCA visualization figure:")
    print(f"1) {overlay_path}")
    print(
        f"ORG points: {org_points.shape[0]}, SMALL points: {small_points.shape[0]}, "
        f"explained variance: PC1={explained_ratio[0] * 100:.2f}%, PC2={explained_ratio[1] * 100:.2f}%"
    )
    if VISUAL_SEPARATION_MODE:
        print(
            f"Center distance (for visualization): before={base_gap:.4f}, after={new_gap:.4f}, "
            f"push_multiplier={CENTER_PUSH_MULTIPLIER}"
        )


if __name__ == "__main__":
    main()
