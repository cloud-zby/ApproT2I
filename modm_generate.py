from tqdm import tqdm
import pickle as pkl
import torch
import os
from PIL import Image
import json
from torchvision import transforms
import argparse
import gc
from config import Config
import glob

SDPipeline = Config.get_pipeline_class()

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_splits', type=int, default=1, help='Number of splits used in traj_*.pkl files')
args = parser.parse_args()

NUM_OF_SPLITS = args.num_of_splits
REUSE_INFO_PATH = Config.REUSE_INFO_PATH
TOTAL_INFERENCE_STEPS = Config.GENERATION_TOTAL_INFERENCE_STEPS
BEGIN_TIMESTEP = Config.GENERATION_BEGIN_TIMESTEP

pipe = SDPipeline.from_pretrained(Config.MODEL_CKPT_PATH, torch_dtype=torch.float16)
if Config.PIPELINE_CLASS_NAME == 'FLUX2KleinPipeline':
    print("Using FLUX.2-klein-base-9B with model CPU offload to reduce GPU memory usage in MoDM.")
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to('cuda')
pipe.set_progress_bar_config(disable=True)

# 加载待被复用的轨迹
metas = []
for split in range(NUM_OF_SPLITS):
    metainfo_path = os.path.join(Config.TRAJ_ALL_META_PKL_PATH_TEMPLATE, f"meta_{split}.pkl")
    meta = pkl.load(open(metainfo_path, 'rb'))
    metas += meta

num_requests = len(metas)

# 加载复用信息
with open(REUSE_INFO_PATH, 'r') as f:
    reuse_infos = json.load(f)

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

if Config.MoDM_DO_NORM:
    target_mean, target_std = compute_org_latent_stats(
            org_dir=Config.TRAJ_ALL_SAVE_DIR,
            step=Config.GENERATION_BEGIN_TIMESTEP,
    )
else:
    target_mean, target_std = None, None

progress_bar = tqdm(range(num_requests), desc='Generating images')
for i in progress_bar:
    torch.cuda.empty_cache()
    gc.collect()
    sample_id = str(metas[i]['uuid'])
    prompt = metas[i]['prompt']
    # 读取profile得到的跳步信息
    reuse_img_idx, prompt_similarity, _, curr_prompt, reuse_prompt = reuse_infos[sample_id]
    if reuse_img_idx is None:
        print(f"Skipping image_id={sample_id} as no reuse info found.")
        continue
    output_path = f"{Config.MODM_OUTPUT_DIR}/{sample_id}_{BEGIN_TIMESTEP}.png"
    if os.path.exists(output_path):
        continue

    # 按需加载原始图像
    org_img_path = f"{Config.TRAJ_ALL_SAVE_DIR}/{reuse_img_idx}.png"
    org_img = Image.open(org_img_path).convert('RGB')
    generator = torch.Generator("cuda").manual_seed(Config.GENERATOR_SEED)
    latent = pipe.modm_image_to_latent(
        image=transforms.ToTensor()(org_img).unsqueeze(0).to(dtype=torch.float16, device="cuda"), 
        step=BEGIN_TIMESTEP,
        total_inference_steps=TOTAL_INFERENCE_STEPS,
        generator=generator,
    )
    # 将latent分布修正为target_mean, target_std
    if Config.MoDM_DO_NORM:
        latent_dtype = latent.dtype
        latent = latent.float()
        latent_mean = latent.mean().item()
        latent_std = latent.std().item()
        latent = (latent - latent_mean) / (latent_std + 1e-6) * target_std + target_mean
        latent = latent.to(dtype=latent_dtype)
    
    # 生成
    generator = torch.Generator("cuda").manual_seed(Config.GENERATOR_SEED)
    img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=BEGIN_TIMESTEP, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
    os.makedirs(Config.MODM_OUTPUT_DIR, exist_ok=True)
    img.save(output_path)

    progress_bar.set_postfix_str(
        f"Similarity={prompt_similarity:.2f}"
    )


