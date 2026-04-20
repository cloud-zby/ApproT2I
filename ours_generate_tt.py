from tqdm import tqdm
import pickle as pkl
import torch
import os
import glob
import json
import argparse
import gc
from config import Config

SDPipeline = Config.get_pipeline_class()

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_splits', type=int, default=1, help='Number of splits used in traj_*.pkl files')
args = parser.parse_args()

NUM_OF_SPLITS = args.num_of_splits
DTYPE = torch.float16
REUSE_INFO_BY_PROMPT_PATH = Config.REUSE_INFO_BY_PROMPT_PATH
PRELIMINARY_STEPS = Config.PRELIMINARY_STEPS
TOTAL_INFERENCE_STEPS = Config.GENERATION_TOTAL_INFERENCE_STEPS
BEGIN_TIMESTEP = Config.GENERATION_BEGIN_TIMESTEP
OURS_SELF_WEIGHT = Config.OURS_SELF_WEIGHT
REFERENCE_STEP = Config.GENERATION_BEGIN_TIMESTEP

pipe = SDPipeline.from_pretrained(Config.MODEL_CKPT_PATH, torch_dtype=DTYPE)
pipe = pipe.to('cuda')
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


def parse_top1_by_prompt(reuse_infos: dict, sample_id: str):
    sample_info = reuse_infos.get(sample_id)
    if not isinstance(sample_info, dict):
        return None, None, None, None

    top_k = sample_info.get('top_k', [])
    if not isinstance(top_k, list) or len(top_k) == 0:
        return None, None, sample_info.get('query_prompt'), None

    top1 = top_k[0]
    if not isinstance(top1, dict):
        return None, None, sample_info.get('query_prompt'), None

    return (
        top1.get('reuse_img_id'),
        top1.get('similarity'),
        sample_info.get('query_prompt'),
        top1.get('candidate_prompt'),
    )

# 加载待被复用的轨迹
metas = []
for split in range(NUM_OF_SPLITS):
    metainfo_path = os.path.join(Config.TRAJ_ALL_META_PKL_PATH_TEMPLATE, f"meta_{split}.pkl")
    meta = pkl.load(open(metainfo_path, 'rb'))
    metas += meta
num_requests = len(metas)
print(f"Total {num_requests} images to generate.")

# 加载prompt检索复用信息（只使用top1）
with open(REUSE_INFO_BY_PROMPT_PATH, 'r') as f:
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
    reuse_img_idx, prompt_similarity, curr_prompt, reuse_prompt = parse_top1_by_prompt(reuse_infos, sample_id)
    if BEGIN_TIMESTEP is None or reuse_img_idx is None:
        print(f"Skipping image_id={sample_id} as no reuse info found.")
        continue
    output_path = f"{Config.OURS_T_T_OUTPUT_DIR}/{sample_id}_{BEGIN_TIMESTEP}.png"
    if os.path.exists(output_path):
        continue

    # 按需加载latent
    with open(f"{Config.TRAJ_ALL_SAVE_DIR}/{reuse_img_idx}.pkl", "rb") as f:
        reuse_traj = pkl.load(f)
    begin_latent = torch.tensor(reuse_traj[BEGIN_TIMESTEP - 1], dtype=DTYPE).cuda()
    with open(f"{Config.SMALL_STEPS_IMAGE_DIR}/{sample_id}.pkl", "rb") as f:
        self_traj = pkl.load(f)
    assert len(self_traj) >= PRELIMINARY_STEPS, f"self_traj length {len(self_traj)} less than PRELIMINARY_STEPS {PRELIMINARY_STEPS}, something wrong!"
    self_latent = torch.tensor(self_traj[PRELIMINARY_STEPS - 1], dtype=DTYPE).cuda()
    assert begin_latent.shape == self_latent.shape, f"Shape mismatch: begin_latent {begin_latent.shape} vs self_latent {self_latent.shape}, something wrong!"

    # Match self_latent distribution to global org latent statistics.
    eps = 1e-6
    self_stats = self_latent.float()
    self_mean = self_stats.mean()
    self_std = self_stats.std(unbiased=False)
    self_latent_matched = ((self_stats - self_mean) / (self_std + eps)) * (target_std + eps) + target_mean
    self_latent_matched = self_latent_matched.to(dtype=DTYPE)

    latent = self_latent_matched * OURS_SELF_WEIGHT + begin_latent * (1 - OURS_SELF_WEIGHT)
    # latent = self_latent_matched
    # latent = self_latent
    # latent = begin_latent

    # print(f"Latent stats: mean={latent.mean().item():.4f}, std={latent.std().item():.4f}")
    # latent = (latent - latent.mean()) / (latent.std() + 1e-8)
    
    generator = torch.Generator("cuda").manual_seed(Config.GENERATOR_SEED)
    img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=BEGIN_TIMESTEP, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
    os.makedirs(Config.OURS_T_T_OUTPUT_DIR, exist_ok=True)
    img.save(output_path)

    sim_text = f"{prompt_similarity:.2f}" if isinstance(prompt_similarity, (int, float)) else "N/A"
    progress_bar.set_postfix_str(
        f"Similarity={sim_text}"
    )


