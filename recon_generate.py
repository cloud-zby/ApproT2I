from tqdm import tqdm
import pickle as pkl
import torch
import os
import numpy as np
import json
import argparse
import gc
from config import Config

SDPipeline = Config.get_pipeline_class()

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_splits', type=int, default=1, help='Number of splits used in traj_*.pkl files')
args = parser.parse_args()

NUM_OF_SPLITS = args.num_of_splits
REUSE_INFO_PATH_BY_PROMPT = Config.REUSE_INFO_BY_PROMPT_PATH
TOTAL_INFERENCE_STEPS = Config.GENERATION_TOTAL_INFERENCE_STEPS
BEGIN_TIMESTEP = Config.GENERATION_BEGIN_TIMESTEP
TOP_K = Config.TOP_K

pipe = SDPipeline.from_pretrained(Config.MODEL_CKPT_PATH, torch_dtype=torch.float16)
pipe = pipe.to('cuda')
pipe.set_progress_bar_config(disable=True)

# 加载待被复用的轨迹
metas = []
for split in range(NUM_OF_SPLITS):
    meta = pkl.load(open(f'meta_{split}.pkl', 'rb'))
    metas += meta
num_requests = len(metas)

# 加载复用信息
with open(REUSE_INFO_PATH_BY_PROMPT, 'r') as f:
    reuse_infos = json.load(f)

progress_bar = tqdm(range(num_requests), desc='Generating images')
for i in progress_bar:
    torch.cuda.empty_cache()
    gc.collect()
    sample_id = str(metas[i]['uuid'])
    prompt = metas[i]['prompt']
    curr_reuse_info = reuse_infos.get(sample_id, {})
    top_k_matches = curr_reuse_info.get('top_k', []) if isinstance(curr_reuse_info, dict) else []

    if not top_k_matches:
        print(f"Skipping image_id={sample_id} as no reuse info found.")
        continue

    # 先按相似度降序，再取最多TOP_K个候选；命中不足TOP_K时按实际数量处理
    sorted_matches = sorted(
        top_k_matches,
        key=lambda x: float(x.get('similarity', float('-inf'))),
        reverse=True
    )
    selected_matches = sorted_matches[:TOP_K]
    valid_matches = []
    for match in selected_matches:
        reuse_img_id = match.get('reuse_img_id')
        similarity = match.get('similarity')
        if reuse_img_id is None or similarity is None:
            continue
        try:
            valid_matches.append((reuse_img_id, float(similarity)))
        except (TypeError, ValueError):
            continue

    if not valid_matches:
        print(f"Skipping image_id={sample_id} as no valid reuse candidate found.")
        continue

    # 归一化到[0,1]
    similarities = np.array([sim for _, sim in valid_matches], dtype=np.float32)
    sim_min = float(similarities.min())
    sim_max = float(similarities.max())
    if sim_max > sim_min:
        norm_weights = (similarities - sim_min) / (sim_max - sim_min)
    else:
        # 只有一个候选或多个候选相似度相同，退化为等权
        norm_weights = np.ones_like(similarities, dtype=np.float32)

    # 归一化权重和为1
    norm_weights = 0.6 * norm_weights + 0.4
    weight_sum = float(norm_weights.sum())
    if weight_sum <= 0:
        weights = np.ones_like(norm_weights, dtype=np.float32) / float(len(norm_weights))
    else:
        weights = norm_weights / weight_sum
    print(f"weights: {weights}, sims: {similarities}")

    best_match = max(valid_matches, key=lambda x: x[1])
    reuse_img_idx = best_match[0]
    prompt_similarity = best_match[1]
    begin_timestep = BEGIN_TIMESTEP

    output_path = f"{Config.RECON_OUTPUT_DIR}/{sample_id}_{begin_timestep}.png"
    if os.path.exists(output_path):
        continue

    # 按需加载候选latent并按权重加权求和（候选数可为1~TOP_K）
    weighted_latent = None
    for (reuse_img_id, _), w in zip(valid_matches, weights.tolist()):
        with open(f"{Config.TRAJ_ALL_SAVE_DIR}/{reuse_img_id}.pkl", "rb") as f:
            reuse_traj = pkl.load(f)
        curr_latent = torch.tensor(reuse_traj[begin_timestep - 1], dtype=torch.float16).cuda()
        if weighted_latent is None:
            weighted_latent = curr_latent * float(w)
        else:
            weighted_latent = weighted_latent + curr_latent * float(w)
    if weighted_latent is None:
        print(f"Skipping image_id={sample_id} as weighted latent is empty.")
        continue
    
    latent = weighted_latent + torch.randn_like(weighted_latent).to(weighted_latent.dtype).to(weighted_latent.device) * Config.ADD_NOISE_SCALE

    generator = torch.Generator("cuda").manual_seed(Config.GENERATOR_SEED)
    img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=begin_timestep, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
    os.makedirs(Config.RECON_OUTPUT_DIR, exist_ok=True)
    img.save(output_path)

    progress_bar.set_postfix_str(
        f"Similarity={prompt_similarity:.2f}"
    )


