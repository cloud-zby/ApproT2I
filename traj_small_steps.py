import os
import json
import random
import tqdm
import numpy
import pickle
import os
import random
import tqdm
import numpy
import pickle
from load_dataset import load_dataset
import torch
import argparse
from config import Config

SDPipeline = Config.get_pipeline_class()

if Config.PIPELINE_CLASS_NAME != "FLUX2KleinPipeline":
    SMALL_STEPS_INFERENCE_STEPS_FLUX = Config.SMALL_STEPS_INFERENCE_STEPS_FLUX
elif Config.PIPELINE_CLASS_NAME == "SD2.1Pipeline" or Config.PIPELINE_CLASS_NAME == "SDXLPipeline":
    SMALL_STEPS_INFERENCE_STEPS_FLUX = None
else:
    raise ValueError(f"Unsupported PIPELINE_CLASS_NAME={Config.PIPELINE_CLASS_NAME} for SMALL_STEPS_INFERENCE_STEPS_FLUX setting.")

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=0, help='数据分片编号')
parser.add_argument('--step', type=int, default=1000, help='每个分片的样本数')
args = parser.parse_args()
split = args.split
STEP = args.step
devices = f"cuda:{split % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"


pipe = SDPipeline.from_pretrained(Config.MODEL_CKPT_PATH, torch_dtype=torch.float16)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
pipe = pipe.to(devices)
# pipe.enable_model_cpu_offload()

def gen_traj(meta, pipe, save_dir=Config.SMALL_STEPS_IMAGE_DIR):
    prompt = meta['prompt']
    image_path = os.path.join(save_dir, f"{meta['uuid']}.png")
    latent_path = os.path.join(save_dir, f"{meta['uuid']}.pkl")
    if os.path.exists(image_path) and os.path.exists(latent_path):
        with open(latent_path, 'rb') as f:
            traj = pickle.load(f)
        return traj

    generator = torch.Generator(devices).manual_seed(Config.GENERATOR_SEED)

    def collect_latents(step, timestep, latents, traj):
        traj.append(latents.cpu().numpy())

    traj = []
    images = pipe(
        prompt=prompt, 
        callback_steps=1, 
        callback=lambda step, timestep, latents: collect_latents(step, timestep, latents, traj), 
        guidance_scale=Config.GUIDANCE_SCALE, 
        generator=generator, 
        early_steps_timesteps=Config.TIMESTEPS_FOR_SMALL_STEPS,
        num_inference_steps=SMALL_STEPS_INFERENCE_STEPS_FLUX,
    ).images

    os.makedirs(save_dir, exist_ok=True)
    images[0].save(image_path)

    # 保存latent为单独的pkl文件
    with open(latent_path, 'wb') as f:
        pickle.dump(traj, f)

    return traj

# 加载DiffusionDB数据集
df_unique = load_dataset(dataset_name=Config.DATASET_NAME)
ids = df_unique.index.tolist()

random.seed(Config.DATA_SHUFFLE_SEED)
random.shuffle(ids)

print(f'split = {split}')
metainfo = []
pipe.set_progress_bar_config(disable=True)
print(f'interval = {STEP * split}, {STEP * (split + 1)}')
with torch.no_grad():
    for idx in tqdm.tqdm(ids[STEP * split: STEP * (split + 1)], desc=f'Generating split {split}'):
        meta = df_unique.loc[idx]
        meta_dict = meta.to_dict()
        gen_traj(meta_dict, pipe, save_dir=Config.SMALL_STEPS_IMAGE_DIR)
        metainfo.append(meta_dict)

print(f"Generated {len(metainfo)} images in split {split}.")
# 保存所有meta信息
pickle.dump(metainfo, open(f'meta_{split}.pkl', 'wb'))