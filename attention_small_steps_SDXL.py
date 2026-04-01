import os
import random
import tqdm
import math
import numpy as np
import pickle
from collections import defaultdict
from load_dataset import load_dataset
import torch
import argparse
from config import Config

def run_attention_small_steps_sdxl(SPLIT, STEP):
    if Config.PIPELINE_CLASS_NAME != 'SDXLPipeline':
        raise ValueError(
            f"attention_small_steps_SDXL.py requires PIPELINE_CLASS_NAME='SDXLPipeline', got {Config.PIPELINE_CLASS_NAME}"
        )

    SDPipeline = Config.get_pipeline_class()

    devices = f"cuda:{SPLIT % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"

    pipe = SDPipeline.from_pretrained(Config.MODEL_CKPT_PATH, torch_dtype=torch.float16)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe = pipe.to(devices)

    cross_attention_query_scores = defaultdict(list)


    def find_sdxl_cross_attn_processor_names(unet: torch.nn.Module) -> set[str]:
        cross_attn_processor_names: set[str] = set()
        for module_name, module in unet.named_modules():
            if not module_name.endswith("attn2"):
                continue
            if not hasattr(module, "to_q") or not hasattr(module, "to_k"):
                continue

            q_in_features = getattr(module.to_q, "in_features", None)
            k_in_features = getattr(module.to_k, "in_features", None)

            # In SDXL cross-attention, query comes from image latent tokens while key/value come from text tokens.
            # This is reflected by mismatched input dims: to_q(in=image_dim) vs to_k(in=text_dim=2048).
            if q_in_features is not None and k_in_features is not None and q_in_features != k_in_features:
                cross_attn_processor_names.add(f"{module_name}.processor")

        return cross_attn_processor_names


    class CaptureAttnProcessor(torch.nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.name = name

        def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            temb: torch.Tensor | None = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            qk_product = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
            if attention_mask is not None:
                qk_product = qk_product + attention_mask
            attn_probs = torch.softmax(qk_product, dim=-1)

            if attn_probs.shape[0] > 1 and attn_probs.shape[0] % 2 == 0:
                batch_idx = attn_probs.shape[0] // 2
            else:
                batch_idx = 0
            attn_mean_head = attn_probs[batch_idx].mean(dim=0)  # (query, key)
            # q axis = image latent tokens, k axis = text tokens.
            query_meankey = attn_mean_head.mean(dim=1)
            cross_attention_query_scores[self.name].append(query_meankey.detach().to(dtype=torch.float32))

            hidden_states = torch.matmul(attn_probs, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states


    def infer_query_hw(query_dim: int, image_h: int, image_w: int) -> tuple[int, int]:
        side = int(round(math.sqrt(query_dim)))
        if side * side == query_dim:
            return side, side

        target_ratio = image_h / max(image_w, 1)
        best_h, best_w = 1, query_dim
        best_err = float("inf")
        max_factor = int(math.sqrt(query_dim))
        for h in range(1, max_factor + 1):
            if query_dim % h != 0:
                continue
            w = query_dim // h
            ratio_err = abs((h / w) - target_ratio)
            if ratio_err < best_err:
                best_err = ratio_err
                best_h, best_w = h, w
        return best_h, best_w


    def query_scores_to_pixel_matrix(
        query_scores: torch.Tensor,
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        query_scores = query_scores.to(dtype=torch.float32)
        query_dim = int(query_scores.shape[0])
        q_h, q_w = infer_query_hw(query_dim, image_h, image_w)
        query_map = query_scores.reshape(q_h, q_w)

        query_map_t = query_map[None, None, :, :]
        pixel_map = torch.nn.functional.interpolate(
            query_map_t,
            size=(image_h, image_w),
            mode="nearest",
        )[0, 0]

        v_min = pixel_map.min()
        v_max = pixel_map.max()
        if (v_max - v_min).item() > 1e-12:
            pixel_map = (pixel_map - v_min) / (v_max - v_min)
        else:
            pixel_map = torch.zeros_like(pixel_map)
        return pixel_map.to(dtype=torch.float32)


    new_attn_procs = {}
    cross_attn_proc_names = find_sdxl_cross_attn_processor_names(pipe.unet)
    if len(cross_attn_proc_names) == 0:
        raise RuntimeError("No SDXL cross-attention processors were detected in UNet.")

    for name, proc in pipe.unet.attn_processors.items():
        if name in cross_attn_proc_names:
            new_attn_procs[name] = CaptureAttnProcessor(name)
        else:
            new_attn_procs[name] = proc
    pipe.unet.set_attn_processor(new_attn_procs)
    print(f"Captured {len(cross_attn_proc_names)} SDXL cross-attention processors.")


    def gen_attention_masks(meta, pipe, image_dir=Config.SMALL_STEPS_IMAGE_DIR, attn_dir=Config.SMALL_STEPS_ATTN_DIR):
        prompt = meta['prompt']
        image_path = os.path.join(image_dir, f"{meta['uuid']}.png")
        attn_path = os.path.join(attn_dir, f"{meta['uuid']}_attn_matrix.pkl")
        if os.path.exists(image_path) and os.path.exists(attn_path):
            return

        generator = torch.Generator(devices).manual_seed(Config.GENERATOR_SEED)

        cross_attention_query_scores.clear()
        images = pipe(
            prompt=prompt,
            guidance_scale=Config.GUIDANCE_SCALE,
            generator=generator,
            early_steps_timesteps=Config.TIMESTEPS_FOR_SMALL_STEPS,
        ).images

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(attn_dir, exist_ok=True)
        images[0].save(image_path)

        image_w, image_h = images[0].size
        step_avg_matrices = []
        if len(cross_attention_query_scores) > 0:
            max_steps = max(len(step_query_scores) for step_query_scores in cross_attention_query_scores.values())
            for step_idx in range(max_steps):
                step_pixel_masks = []
                for step_query_scores in cross_attention_query_scores.values():
                    if step_idx < len(step_query_scores):
                        pixel_mask = query_scores_to_pixel_matrix(step_query_scores[step_idx], image_h, image_w)
                        step_pixel_masks.append(pixel_mask)
                if len(step_pixel_masks) > 0:
                    step_avg = torch.mean(torch.stack(step_pixel_masks, dim=0), dim=0)
                    step_avg_matrices.append(step_avg.detach().cpu().numpy().astype(np.float32))

        with open(attn_path, 'wb') as f:
            pickle.dump(step_avg_matrices, f)

        return


    # 加载数据集并统一随机种子
    df_unique = load_dataset(dataset_name=Config.DATASET_NAME)
    ids = df_unique.index.tolist()
    random.seed(Config.DATA_SHUFFLE_SEED)
    random.shuffle(ids)

    print(f'split = {SPLIT}')
    metainfo = []
    pipe.set_progress_bar_config(disable=True)
    print(f'interval = {STEP * SPLIT}, {STEP * (SPLIT + 1)}')
    with torch.no_grad():
        for idx in tqdm.tqdm(ids[STEP * SPLIT: STEP * (SPLIT + 1)], desc=f'Generating split {SPLIT}'):
            meta = df_unique.loc[idx]
            meta_dict = meta.to_dict()
            gen_attention_masks(
                meta_dict,
                pipe,
                image_dir=Config.SMALL_STEPS_IMAGE_DIR,
                attn_dir=Config.SMALL_STEPS_ATTN_DIR,
            )
            metainfo.append(meta_dict)

    print(f"Generated {len(metainfo)} images in split {SPLIT}.")
