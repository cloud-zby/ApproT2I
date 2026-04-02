import os
import random
import tqdm
import math
import numpy as np
import pickle
from collections import defaultdict
from pipeline import FLUX2Pipeline
from load_dataset import load_dataset
import torch
import argparse
from config import Config
from PIL import Image
from diffusers.models.transformers.transformer_flux2 import _get_qkv_projections, apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn

VISUALIZE_ATTN_HEATMAP = False

def run_attention_small_steps_flux(SPLIT, STEP):

    if Config.PIPELINE_CLASS_NAME != 'FLUX2KleinPipeline':
        raise ValueError(
            f"attention_small_steps_flux.py requires PIPELINE_CLASS_NAME='FLUX2KleinPipeline', got {Config.PIPELINE_CLASS_NAME}"
        )

    devices = f"cuda:{SPLIT % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"

    pipe = FLUX2Pipeline.from_pretrained(Config.MODEL_CKPT_PATHS["FLUX2KleinPipeline"], torch_dtype=torch.bfloat16)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe = pipe.to(devices)
    # pipe.enable_model_cpu_offload()

    cross_attention_query_scores = defaultdict(list)

    class CaptureFlux2AttnProcessor:
        _attention_backend = None
        _parallel_config = None

        def __init__(self, name: str):
            self.name = name

        def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            image_rotary_emb: torch.Tensor | None = None,
        ):
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            text_seq_len = None
            if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
                text_seq_len = encoder_hidden_states.shape[1]
                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
                encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

                encoder_query = attn.norm_added_q(encoder_query)
                encoder_key = attn.norm_added_k(encoder_key)

                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            if text_seq_len is not None and text_seq_len > 0:
                q = query.permute(0, 2, 1, 3)
                k = key.permute(0, 2, 1, 3)
                qk_product = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
                if attention_mask is not None:
                    qk_product = qk_product + attention_mask
                attn_probs = torch.softmax(qk_product, dim=-1)

                batch_idx = 0
                attn_mean_head = attn_probs[batch_idx].mean(dim=0)  # (query, key)
                if attn_mean_head.shape[0] > text_seq_len and attn_mean_head.shape[1] >= text_seq_len:
                    cross_attn = attn_mean_head[text_seq_len:, :text_seq_len]
                    query_meankey = cross_attn.mean(dim=1)
                    cross_attention_query_scores[self.name].append(query_meankey.detach().to(dtype=torch.float32))

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                    [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
                )
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            return hidden_states


    def find_flux_cross_attn_processor_names(transformer: torch.nn.Module) -> set[str]:
        cross_attn_processor_names: set[str] = set()
        for name in transformer.attn_processors.keys():
            if "single_transformer_blocks" in name:
                continue
            if "transformer_blocks" in name and name.endswith("attn.processor"):
                cross_attn_processor_names.add(name)
        return cross_attn_processor_names


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


    def save_step_attention_heatmaps(
        step_avg_matrices: list[np.ndarray],
        output_dir: str,
        base_name: str,
    ) -> None:
        if len(step_avg_matrices) == 0:
            return

        os.makedirs(output_dir, exist_ok=True)

        for step_idx, step_mask in enumerate(step_avg_matrices):
            if step_mask.ndim != 2:
                continue
            mask = np.clip(step_mask, 0.0, 1.0)
            heatmap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
            heatmap[..., 0] = mask * 255.0

            heatmap_img = Image.fromarray(np.clip(heatmap, 0.0, 255.0).astype(np.uint8))
            heatmap_path = os.path.join(output_dir, f"{base_name}_attn_step_{step_idx:03d}.png")
            heatmap_img.save(heatmap_path)


    cross_attn_proc_names = find_flux_cross_attn_processor_names(pipe.transformer)
    if len(cross_attn_proc_names) == 0:
        raise RuntimeError("No Flux2 cross-attention processors were detected in transformer_blocks.")

    new_attn_procs = {}
    for name, proc in pipe.transformer.attn_processors.items():
        if name in cross_attn_proc_names:
            new_attn_procs[name] = CaptureFlux2AttnProcessor(name)
        else:
            new_attn_procs[name] = proc
    pipe.transformer.set_attn_processor(new_attn_procs)
    print(f"Captured {len(cross_attn_proc_names)} Flux2 cross-attention processors.")


    def gen_attention_masks(meta, pipe, image_dir=Config.SMALL_STEPS_IMAGE_DIR, attn_dir=Config.SMALL_STEPS_ATTN_DIR):
        prompt = meta["prompt"]
        image_path = os.path.join(image_dir, f"{meta['uuid']}.png")
        attn_path = os.path.join(attn_dir, f"{meta['uuid']}_attn_matrix.pkl")
        if os.path.exists(image_path) and os.path.exists(attn_path):
            return

        generator = torch.Generator(devices).manual_seed(Config.GENERATOR_SEED)

        cross_attention_query_scores.clear()

        early_steps_timesteps = Config.TIMESTEPS_FOR_SMALL_STEPS

        images = pipe(
            prompt=prompt,
            guidance_scale=Config.GUIDANCE_SCALE,
            generator=generator,
            num_inference_steps=10,
            early_steps_timesteps=early_steps_timesteps,
        ).images

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(attn_dir, exist_ok=True)
        images[0].save(image_path)

        image_w, image_h = images[0].size
        step_avg_matrices = []
        if len(cross_attention_query_scores) > 0:
            expected_steps = (
                len(early_steps_timesteps) if early_steps_timesteps is not None else TOTAL_INFERENCE_STEPS
            )
            cleaned_scores = {}
            for name, step_query_scores in cross_attention_query_scores.items():
                if len(step_query_scores) == expected_steps * 2:
                    cleaned_scores[name] = step_query_scores[0::2]
                else:
                    cleaned_scores[name] = step_query_scores

            max_steps = max(len(step_query_scores) for step_query_scores in cleaned_scores.values())
            for step_idx in range(max_steps):
                step_pixel_masks = []
                for step_query_scores in cleaned_scores.values():
                    if step_idx < len(step_query_scores):
                        pixel_mask = query_scores_to_pixel_matrix(step_query_scores[step_idx], image_h, image_w)
                        step_pixel_masks.append(pixel_mask)
                if len(step_pixel_masks) > 0:
                    step_avg = torch.mean(torch.stack(step_pixel_masks, dim=0), dim=0)
                    step_avg_matrices.append(step_avg.detach().cpu().numpy().astype(np.float32))

        with open(attn_path, "wb") as f:
            pickle.dump(step_avg_matrices, f)

        if VISUALIZE_ATTN_HEATMAP:
            heatmap_dir = os.path.join(attn_dir, "heatmaps")
            save_step_attention_heatmaps(step_avg_matrices, heatmap_dir, meta["uuid"])

        return


    # 加载数据集
    df_unique = load_dataset(dataset_name=Config.DATASET_NAME)
    ids = df_unique.index.tolist()

    random.seed(Config.DATA_SHUFFLE_SEED)
    random.shuffle(ids)

    print(f"split = {SPLIT}")
    metainfo = []
    pipe.set_progress_bar_config(disable=True)
    print(f"interval = {STEP * SPLIT}, {STEP * (SPLIT + 1)}")
    with torch.no_grad():
        for idx in tqdm.tqdm(ids[STEP * SPLIT: STEP * (SPLIT + 1)], desc=f"Generating split {SPLIT}"):
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
