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
    
    # 生成
    generator = torch.Generator("cuda").manual_seed(Config.GENERATOR_SEED)
    img = pipe(prompt=prompt, head_start_latents=latent, head_start_step=BEGIN_TIMESTEP, guidance_scale=Config.GUIDANCE_SCALE, generator=generator, num_inference_steps=TOTAL_INFERENCE_STEPS).images[0]
    os.makedirs(Config.MODM_OUTPUT_DIR, exist_ok=True)
    img.save(output_path)

    progress_bar.set_postfix_str(
        f"Similarity={prompt_similarity:.2f}"
    )


