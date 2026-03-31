from tqdm import tqdm
import pickle as pkl
import torch
import os
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
    best_match = max(top_k_matches, key=lambda x: float(x.get('similarity', float('-inf'))))
    reuse_img_idx = best_match.get('reuse_img_id')
    prompt_similarity = float(best_match.get('similarity', 0.0))
    begin_timestep = BEGIN_TIMESTEP
    if reuse_img_idx is None:
        print(f"Skipping image_id={sample_id} as no valid reuse candidate found.")
        continue

    output_path = f"{Config.NIRVANA_OUTPUT_DIR}/{sample_id}_{begin_timestep}.png"
    if os.path.exists(output_path):
        continue

    # 按需加载latent
    with open(f"{Config.TRAJ_ALL_SAVE_DIR}/{reuse_img_idx}.pkl", "rb") as f:
        reuse_traj = pkl.load(f)
    begin_latent = torch.tensor(reuse_traj[begin_timestep - 1], dtype=torch.float16).cuda()
    
    latent = begin_latent

    generator = torch.Generator("cuda").manual_seed(Config.GENERATOR_SEED)
    img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=begin_timestep, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
    os.makedirs(Config.NIRVANA_OUTPUT_DIR, exist_ok=True)
    img.save(output_path)

    progress_bar.set_postfix_str(
        f"Similarity={prompt_similarity:.2f}"
    )


