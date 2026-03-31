from tqdm import tqdm
import pickle as pkl
import torch
import os
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pipe = SDPipeline.from_pretrained(Config.MODEL_CKPT_PATH, torch_dtype=DTYPE)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)


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


# 加载待被复用的轨迹
metas = []
for split in range(NUM_OF_SPLITS):
    meta = pkl.load(open(f'meta_{split}.pkl', 'rb'))
    metas += meta
num_requests = len(metas)
print(f"Total {num_requests} images to generate.")

# 加载复用信息
with open(REUSE_INFO_PATH, 'r') as f:
    reuse_infos = json.load(f)

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
    output_path = f"{Config.OURS_MIX_OUTPUT_DIR}/{sample_id}_{BEGIN_TIMESTEP}.png"
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

    # Match self_latent distribution to begin_latent distribution before fusion.
    eps = 1e-6
    begin_stats = begin_latent.float()
    self_stats = self_latent.float()
    begin_mean = begin_stats.mean()
    begin_std = begin_stats.std(unbiased=False)
    self_mean = self_stats.mean()
    self_std = self_stats.std(unbiased=False)
    self_latent_matched = ((self_stats - self_mean) / (self_std + eps)) * (begin_std + eps) + begin_mean
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
    # 调整token_mask整体偏移量，使得>=0.5的占比在55%~60%之间（类似于二分搜索，以减少迭代次数）
    # def adjust_token_mask(mask, lower_ratio, upper_ratio, max_iter, shift):
    #     mask = mask.copy()
    #     for iters in range(max_iter):
    #         ratio = np.mean(mask >= 0.5)
    #         if lower_ratio <= ratio <= upper_ratio:
    #             print(f"Token mask adjustment converged at iteration {iters} within [{lower_ratio}, {upper_ratio}]")
    #             break
    #         # 计算需要加的偏移量
    #         if ratio < lower_ratio:
    #             # 如果添加当前偏移，则会加太多了
    #             if np.mean(mask + shift >= 0.5) > upper_ratio:
    #                 shift = shift / 2.0  # 减小步长
    #             mask += shift
    #         else:
    #             # 如果减去当前偏移，则会减太多了
    #             if np.mean(mask - shift >= 0.5) < lower_ratio:  
    #                 shift = shift / 2.0  # 减小步长
    #             mask -= shift
    #     return np.clip(mask, 0.10, 0.90)
    # token_mask = adjust_token_mask(mask=token_mask, lower_ratio=0.55, upper_ratio=0.60, max_iter=15, shift=0.03)
    
    # 统计token_mask中大于X的元素个数
    # num_gt_04 = np.sum(token_mask >= 0.45) / token_mask.size
    # num_gt_05 = np.sum(token_mask >= 0.50) / token_mask.size
    # num_gt_06 = np.sum(token_mask >= 0.55) / token_mask.size
    # num_gt_07 = np.sum(token_mask >= 0.6) / token_mask.size
    # num_gt_08 = np.sum(token_mask >= 0.8) / token_mask.size
    # print(f"Token mask stats after normalization: >=0.4: {num_gt_04:.2f}, >=0.5: {num_gt_05:.2f}, >=0.6: {num_gt_06:.2f}, >=0.7: {num_gt_07:.2f}, >=0.8: {num_gt_08:.2f}")

    # 加权融合
    token_mask_torch = token_mask.unsqueeze(0).unsqueeze(0)   # 形状扩展为(1, 1, H, W)以便广播
    latent = self_latent_matched * token_mask_torch + begin_latent * (1 - token_mask_torch) # 加权融合
    # latent = self_latent * 0.6 + begin_latent * 0.4
    
    generator = torch.Generator(device).manual_seed(Config.GENERATOR_SEED)
    img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=BEGIN_TIMESTEP, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
    os.makedirs(Config.OURS_MIX_OUTPUT_DIR, exist_ok=True)
    img.save(output_path)

    progress_bar.set_postfix_str(
        f"Similarity={prompt_similarity:.2f}"
    )


