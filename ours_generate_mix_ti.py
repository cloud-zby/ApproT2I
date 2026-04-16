from tqdm import tqdm
import pickle as pkl
import torch
import os
import glob
from PIL import Image
import json
import torch.nn.functional as F
import argparse
import gc
from config import Config

SDPipeline = Config.get_pipeline_class()

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_splits', type=int, default=1, help='Number of splits used in traj_*.pkl files')
args = parser.parse_args()

NUM_OF_SPLITS = args.num_of_splits
DTYPE = torch.float16
REUSE_INFO_PATH = Config.REUSE_INFO_PATH
PRELIMINARY_STEPS = Config.PRELIMINARY_STEPS
TOTAL_INFERENCE_STEPS = Config.GENERATION_TOTAL_INFERENCE_STEPS
BEGIN_TIMESTEP = Config.GENERATION_BEGIN_TIMESTEP
WEIGHT_MASK_SELF_MIN = Config.OURS_MIX_SELF_WEIGHT_MIN
WEIGHT_MASK_SELF_MAX = Config.OURS_MIX_SELF_WEIGHT_MAX
REFERENCE_STEP = Config.GENERATION_BEGIN_TIMESTEP

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pipe = SDPipeline.from_pretrained(Config.MODEL_CKPT_PATH, torch_dtype=DTYPE)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)


def compute_org_latent_stats(org_dir: str, step: int) -> tuple[float, float]:
    latent_files = sorted(glob.glob(os.path.join(org_dir, "*.pkl")))
    if len(latent_files) == 0:
        raise FileNotFoundError(f"No pkl files found under `{org_dir}`")

    target_idx = step - 1
    mean_sum = 0.0
    var_sum = 0.0
    valid_cnt = 0

    stats_bar = tqdm(latent_files, desc=f"Computing step-{step} latent stats")
    for latent_path in stats_bar:
        with open(latent_path, "rb") as f:
            traj = pkl.load(f)
        if len(traj) <= target_idx:
            continue

        latent = torch.tensor(traj[target_idx], dtype=torch.float32)
        mean_sum += latent.mean().item()
        var_sum += latent.var(unbiased=False).item()
        valid_cnt += 1

        stats_bar.set_postfix_str(f"valid={valid_cnt}")

    if valid_cnt == 0:
        raise ValueError(
            f"No valid trajectories contain step={step}. Please check {org_dir}/*.pkl lengths."
        )

    avg_mean = mean_sum / valid_cnt
    avg_var = var_sum / valid_cnt
    avg_std = avg_var ** 0.5
    print(
        f"Computed from {valid_cnt} files at step={step}: "
        f"avg_mean={avg_mean:.6f}, avg_var={avg_var:.6f}, avg_std={avg_std:.6f}"
    )
    return avg_mean, avg_std


def pixel_mask_to_latent_tokens(pixel_mask: torch.Tensor, latent_shape: torch.Size, vae_scale_factor: int) -> torch.Tensor:
    # pixel_mask: (H, W) torch tensor, device可为cuda/cpu
    assert pixel_mask.ndim == 2, f"Expected 2D mask, got shape={pixel_mask.shape}"
    latent_h, latent_w = int(latent_shape[-2]), int(latent_shape[-1])
    expected_h = latent_h * vae_scale_factor
    expected_w = latent_w * vae_scale_factor
    assert pixel_mask.shape == (expected_h, expected_w), (
        f"Pixel mask shape {pixel_mask.shape} does not match expected {(expected_h, expected_w)} "
        f"from latent shape {(latent_h, latent_w)} and vae_scale_factor={vae_scale_factor}"
    )
    mask_t = pixel_mask[None, None, :, :].to(dtype=torch.float16)
    token_t = F.interpolate(mask_t, size=(latent_h, latent_w), mode="nearest")
    return token_t[0, 0]


def pixel_mask_to_flux_packed_token_weights(
    pixel_mask: torch.Tensor,
    latent: torch.Tensor,
    vae_scale_factor: int,
    patch_size: int = 2,
) -> torch.Tensor:
    # FLUX encode: image(H,W) -> vae latent(H/8,W/8) -> patchify(H/16,W/16) -> pack(seq=H/16*W/16, C)
    assert pixel_mask.ndim == 2, f"Expected 2D mask, got shape={pixel_mask.shape}"
    assert latent.ndim in (2, 3), f"Expected latent ndim in (2,3), got shape={latent.shape}"

    seq_len = int(latent.shape[-2])
    image_h, image_w = int(pixel_mask.shape[0]), int(pixel_mask.shape[1])
    downsample = vae_scale_factor * patch_size

    assert image_h % downsample == 0 and image_w % downsample == 0, (
        f"Pixel mask shape {(image_h, image_w)} must be divisible by {downsample} "
        f"(vae_scale_factor={vae_scale_factor}, patch_size={patch_size})"
    )

    token_h = image_h // downsample
    token_w = image_w // downsample
    expected_seq_len = token_h * token_w
    assert expected_seq_len == seq_len, (
        f"Packed token count mismatch: expected {expected_seq_len} from pixel mask {(image_h, image_w)}, "
        f"but latent seq_len is {seq_len} (latent shape={latent.shape})"
    )

    # Align with FLUX _pack_latents flatten order: row-major flatten over (token_h, token_w).
    mask_t = pixel_mask[None, None, :, :].to(dtype=torch.float32)
    token_grid = F.interpolate(mask_t, size=(token_h, token_w), mode="nearest")[0, 0]
    token_seq = token_grid.reshape(seq_len)

    if latent.ndim == 3:
        return token_seq.view(1, seq_len, 1).expand(int(latent.shape[0]), -1, -1)
    return token_seq.view(seq_len, 1)

def ours_mix_sd_main():
    assert Config.PIPELINE_CLASS_NAME == "SD2.1Pipeline" or Config.PIPELINE_CLASS_NAME == "SDXLPipeline", f"Currently only SD2.1Pipeline and SDXLPipeline are supported, but got {Config.PIPELINE_CLASS_NAME}"
    # 加载待被复用的轨迹
    metas = []
    for split in range(NUM_OF_SPLITS):
        metainfo_path = os.path.join(Config.TRAJ_ALL_META_PKL_PATH_TEMPLATE, f"meta_{split}.pkl")
        meta = pkl.load(open(metainfo_path, 'rb'))
        metas += meta
    num_requests = len(metas)
    print(f"Total {num_requests} images to generate.")

    # 加载复用信息
    with open(REUSE_INFO_PATH, 'r') as f:
        reuse_infos = json.load(f)

    target_mean, target_std = compute_org_latent_stats(
        org_dir=Config.TRAJ_ALL_SAVE_DIR,
        step=REFERENCE_STEP,
    )

    progress_bar = tqdm(range(num_requests), desc='Generating images')
    for i in progress_bar:
        torch.cuda.empty_cache()
        gc.collect()
        sample_id = str(metas[i]['uuid'])
        prompt = metas[i]['prompt']
        reuse_img_idx, prompt_similarity, _, curr_prompt, reuse_prompt = reuse_infos[sample_id]
        if BEGIN_TIMESTEP is None or reuse_img_idx is None:
            print(f"Skipping image_id={sample_id} as no reuse info found.")
            continue
        output_path = f"{Config.OURS_MIX_T_I_OUTPUT_DIR}/{sample_id}_{BEGIN_TIMESTEP}.png"
        if os.path.exists(output_path):
            continue

        # 按需加载latent
        with open(f"{Config.TRAJ_ALL_SAVE_DIR}/{reuse_img_idx}.pkl", "rb") as f:
            reuse_traj = pkl.load(f)
        begin_latent = torch.tensor(reuse_traj[BEGIN_TIMESTEP - 1], dtype=DTYPE).to(device)
        with open(f"{Config.SMALL_STEPS_IMAGE_DIR}/{sample_id}.pkl", "rb") as f:
            self_traj = pkl.load(f)
        assert len(self_traj) >= PRELIMINARY_STEPS, f"self_traj length {len(self_traj)} less than PRELIMINARY_STEPS {PRELIMINARY_STEPS}, something wrong!"
        self_latent = torch.tensor(self_traj[PRELIMINARY_STEPS - 1], dtype=DTYPE).to(device)
        assert begin_latent.shape == self_latent.shape, f"Shape mismatch: begin_latent {begin_latent.shape} vs self_latent {self_latent.shape}, something wrong!"

        # Match self_latent distribution to global org latent statistics before fusion.
        eps = 1e-6
        self_stats = self_latent.float()
        self_mean = self_stats.mean()
        self_std = self_stats.std(unbiased=False)
        self_latent_matched = ((self_stats - self_mean) / (self_std + eps)) * (target_std + eps) + target_mean
        self_latent_matched = self_latent_matched.to(dtype=DTYPE)

        # 加载注意力分数
        attn_path = f"{Config.SMALL_STEPS_ATTN_DIR}/{sample_id}_attn_matrix.pkl"
        assert os.path.exists(attn_path), f"Attention matrix file not found: {attn_path}"
        try:
            with open(attn_path, "rb") as f:
                attn_matrix = pkl.load(f)
        except Exception as e:
            print(f"\033[91mError loading attention matrix from {attn_path}: {e}\033[0m")
            continue
        # 直接读取第 PRELIMINARY_STEPS 步已经跨层平均后的 attention
        assert isinstance(attn_matrix, (list, tuple)), f"Invalid attention format in {attn_path}, expected list/tuple"
        assert len(attn_matrix) > PRELIMINARY_STEPS - 1, (
            f"No valid attention mask found for step {PRELIMINARY_STEPS - 1} in {attn_path}"
        )
        avg_pixel_mask = torch.as_tensor(attn_matrix[PRELIMINARY_STEPS - 1], dtype=torch.float16, device=device)
        # 根据VAE下采样倍率将pixel mask映射回latent token空间
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        # pixel_mask_to_latent_tokens 现在支持torch tensor输入输出，直接在GPU上处理
        token_mask = pixel_mask_to_latent_tokens(avg_pixel_mask, self_latent.shape, vae_scale_factor).to(device=device, dtype=self_latent.dtype)

        # 归一化注意力分数：为避免 float16 在统计函数（如 quantile）上不被支持，
        # 我们将统计计算在 float32 上完成，再把归一化结果转换回 latent 的 dtype。
        token_mask_f32 = token_mask.to(torch.float32)
        q_low = torch.quantile(token_mask_f32, 0.08)
        q_high = torch.quantile(token_mask_f32, 0.92)
        token_mask_f32 = torch.clamp(token_mask_f32, min=q_low, max=q_high)
        mean = token_mask_f32.mean()
        std = token_mask_f32.std(unbiased=False)
        token_mask_f32 = (token_mask_f32 - mean) / (std + 1e-8)
        minv = token_mask_f32.min()
        maxv = token_mask_f32.max()
        token_mask_f32 = (token_mask_f32 - minv) / (maxv - minv + 1e-8) * (WEIGHT_MASK_SELF_MAX - WEIGHT_MASK_SELF_MIN) + WEIGHT_MASK_SELF_MIN
        token_mask_f32 = torch.clamp(token_mask_f32, min=WEIGHT_MASK_SELF_MIN, max=WEIGHT_MASK_SELF_MAX)  # [WEIGHT_MASK_MIN, WEIGHT_MASK_MAX]范围内
        token_mask = token_mask_f32.to(self_latent.dtype)

        # 加权融合
        token_mask_torch = token_mask.unsqueeze(0).unsqueeze(0)   # 形状扩展为(1, 1, H, W)以便广播
        latent = self_latent_matched * token_mask_torch + begin_latent * (1 - token_mask_torch) # 加权融合
        # latent = self_latent * 0.6 + begin_latent * 0.4
        
        generator = torch.Generator(device).manual_seed(Config.GENERATOR_SEED)
        img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=BEGIN_TIMESTEP, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
        os.makedirs(Config.OURS_MIX_T_I_OUTPUT_DIR, exist_ok=True)
        img.save(output_path)

        progress_bar.set_postfix_str(
            f"Similarity={prompt_similarity:.2f}"
        )

def ours_mix_flux_main():
    assert Config.PIPELINE_CLASS_NAME == "FLUX2KleinPipeline", f"Currently only FLUX2KleinPipeline is supported, but got {Config.PIPELINE_CLASS_NAME}"
    # 加载待被复用的轨迹
    metas = []
    for split in range(NUM_OF_SPLITS):
        metainfo_path = os.path.join(Config.TRAJ_ALL_META_PKL_PATH_TEMPLATE, f"meta_{split}.pkl")
        meta = pkl.load(open(metainfo_path, 'rb'))
        metas += meta
    num_requests = len(metas)
    print(f"Total {num_requests} images to generate.")

    # 加载复用信息
    with open(REUSE_INFO_PATH, 'r') as f:
        reuse_infos = json.load(f)

    target_mean, target_std = compute_org_latent_stats(
        org_dir=Config.TRAJ_ALL_SAVE_DIR,
        step=REFERENCE_STEP,
    )

    progress_bar = tqdm(range(num_requests), desc='Generating images')
    for i in progress_bar:
        torch.cuda.empty_cache()
        gc.collect()
        sample_id = str(metas[i]['uuid'])
        prompt = metas[i]['prompt']
        reuse_img_idx, prompt_similarity, _, curr_prompt, reuse_prompt = reuse_infos[sample_id]
        if BEGIN_TIMESTEP is None or reuse_img_idx is None:
            print(f"Skipping image_id={sample_id} as no reuse info found.")
            continue
        output_path = f"{Config.OURS_MIX_T_I_OUTPUT_DIR}/{sample_id}_{BEGIN_TIMESTEP}.png"
        if os.path.exists(output_path):
            continue

        # 按需加载latent
        with open(f"{Config.TRAJ_ALL_SAVE_DIR}/{reuse_img_idx}.pkl", "rb") as f:
            reuse_traj = pkl.load(f)
        begin_latent = torch.tensor(reuse_traj[BEGIN_TIMESTEP - 1], dtype=DTYPE).to(device)
        with open(f"{Config.SMALL_STEPS_IMAGE_DIR}/{sample_id}.pkl", "rb") as f:
            self_traj = pkl.load(f)
        assert len(self_traj) >= PRELIMINARY_STEPS, f"self_traj length {len(self_traj)} less than PRELIMINARY_STEPS {PRELIMINARY_STEPS}, something wrong!"
        self_latent = torch.tensor(self_traj[PRELIMINARY_STEPS - 1], dtype=DTYPE).to(device)
        assert begin_latent.shape == self_latent.shape, f"Shape mismatch: begin_latent {begin_latent.shape} vs self_latent {self_latent.shape}, something wrong!"

        # Match self_latent distribution to global org latent statistics before fusion.
        eps = 1e-6
        self_stats = self_latent.float()
        self_mean = self_stats.mean()
        self_std = self_stats.std(unbiased=False)
        self_latent_matched = ((self_stats - self_mean) / (self_std + eps)) * (target_std + eps) + target_mean
        self_latent_matched = self_latent_matched.to(dtype=DTYPE)

        # 加载注意力分数
        attn_path = f"{Config.SMALL_STEPS_ATTN_DIR}/{sample_id}_attn_matrix.pkl"
        assert os.path.exists(attn_path), f"Attention matrix file not found: {attn_path}"
        try:
            with open(attn_path, "rb") as f:
                attn_matrix = pkl.load(f)
        except Exception as e:
            print(f"\033[91mError loading attention matrix from {attn_path}: {e}\033[0m")
            continue
        # 直接读取第 PRELIMINARY_STEPS 步已经跨层平均后的 attention
        assert isinstance(attn_matrix, (list, tuple)), f"Invalid attention format in {attn_path}, expected list/tuple"
        assert len(attn_matrix) > PRELIMINARY_STEPS - 1, (
            f"No valid attention mask found for step {PRELIMINARY_STEPS - 1} in {attn_path}"
        )
        avg_pixel_mask = torch.as_tensor(attn_matrix[PRELIMINARY_STEPS - 1], dtype=torch.float16, device=device)
        # FLUX 使用 packed latents: (B, seq_len, C) 或 (seq_len, C)。
        # 映射需严格遵循 encode 中的 8x VAE 下采样 + 2x2 patchify + pack。
        vae_scale_factor = getattr(pipe, "vae_scale_factor", 2 ** (len(pipe.vae.config.block_out_channels) - 1))
        token_mask = pixel_mask_to_flux_packed_token_weights(
            avg_pixel_mask,
            self_latent,
            vae_scale_factor=vae_scale_factor,
            patch_size=2,
        ).to(device=device, dtype=self_latent.dtype)

        # 归一化注意力分数：为避免 float16 在统计函数（如 quantile）上不被支持，
        # 我们将统计计算在 float32 上完成，再把归一化结果转换回 latent 的 dtype。
        token_mask_f32 = token_mask.to(torch.float32)
        q_low = torch.quantile(token_mask_f32, 0.08)
        q_high = torch.quantile(token_mask_f32, 0.92)
        token_mask_f32 = torch.clamp(token_mask_f32, min=q_low, max=q_high)
        mean = token_mask_f32.mean()
        std = token_mask_f32.std(unbiased=False)
        token_mask_f32 = (token_mask_f32 - mean) / (std + 1e-8)
        minv = token_mask_f32.min()
        maxv = token_mask_f32.max()
        token_mask_f32 = (token_mask_f32 - minv) / (maxv - minv + 1e-8) * (WEIGHT_MASK_SELF_MAX - WEIGHT_MASK_SELF_MIN) + WEIGHT_MASK_SELF_MIN
        token_mask_f32 = torch.clamp(token_mask_f32, min=WEIGHT_MASK_SELF_MIN, max=WEIGHT_MASK_SELF_MAX)  # [WEIGHT_MASK_MIN, WEIGHT_MASK_MAX]范围内
        token_mask = token_mask_f32.to(self_latent.dtype)

        # 加权融合
        latent = self_latent_matched * token_mask + begin_latent * (1 - token_mask) # 加权融合
        # latent = self_latent * 0.6 + begin_latent * 0.4
        
        generator = torch.Generator(device).manual_seed(Config.GENERATOR_SEED)
        img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=BEGIN_TIMESTEP, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
        os.makedirs(Config.OURS_MIX_T_I_OUTPUT_DIR, exist_ok=True)
        img.save(output_path)

        progress_bar.set_postfix_str(
            f"Similarity={prompt_similarity:.2f}"
        )


if __name__ == "__main__":
    if Config.PIPELINE_CLASS_NAME == "SD2.1Pipeline" or Config.PIPELINE_CLASS_NAME == "SDXLPipeline":
        ours_mix_sd_main()
    elif Config.PIPELINE_CLASS_NAME == "FLUX2KleinPipeline":
        ours_mix_flux_main()
    else:
        raise ValueError(f"Unsupported PIPELINE_CLASS_NAME {Config.PIPELINE_CLASS_NAME} for ours_mix_main.py")

