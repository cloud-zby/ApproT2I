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
REUSE_INFO_PATH = Config.REUSE_INFO_PATH
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
with open(REUSE_INFO_PATH, 'r') as f:
    reuse_infos = json.load(f)

progress_bar = tqdm(range(num_requests), desc='Generating images')
for i in progress_bar:
    torch.cuda.empty_cache()
    gc.collect()
    sample_id = str(metas[i]['uuid'])
    prompt = metas[i]['prompt']
    reuse_img_idx, prompt_similarity, _, curr_prompt, reuse_prompt = reuse_infos[sample_id]
    if reuse_img_idx is None:
        print(f"Skipping image_id={sample_id} as no reuse info found.")
        continue
    output_path = f"{Config.REDI_OUTPUT_DIR}/{sample_id}_{BEGIN_TIMESTEP}.png"
    if os.path.exists(output_path):
        continue

    # 按需加载latent
    with open(f"{Config.TRAJ_ALL_SAVE_DIR}/{reuse_img_idx}.pkl", "rb") as f:
        reuse_traj = pkl.load(f)
    begin_latent = torch.tensor(reuse_traj[BEGIN_TIMESTEP - 1], dtype=torch.float16).cuda()
    
    latent = begin_latent

    generator = torch.Generator("cuda").manual_seed(Config.GENERATOR_SEED)
    img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=BEGIN_TIMESTEP, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
    os.makedirs(Config.REDI_OUTPUT_DIR, exist_ok=True)
    img.save(output_path)

    progress_bar.set_postfix_str(
        f"Similarity={prompt_similarity:.2f}"
    )


