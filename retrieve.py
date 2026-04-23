from tqdm import tqdm
import torch
import os
from PIL import Image
import numpy as np
import json
import time
import pickle as pkl
import faiss
import clip
import random
import argparse
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_splits', type=int, default=1, help='Number of splits used in traj_*.pkl files')
args = parser.parse_args()

NUM_OF_SPLITS = args.num_of_splits
SEED = Config.RETRIEVE_SEED
DISTANCE_METRIC = Config.DISTANCE_METRIC  # 'cosine', 'l2', 'l1'
REUSE_INFO_PATH = Config.REUSE_INFO_PATH

if os.path.exists(REUSE_INFO_PATH):
    print(f"Warning: {REUSE_INFO_PATH} already exists and will be skipped. Please delete it if you want to re-run retrieval.")
else:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 加载CLIP模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    metas = []
    id_to_prompt = {}
    for split in range(NUM_OF_SPLITS):
        meta_path = os.path.join(Config.TRAJ_ALL_META_PKL_PATH_TEMPLATE, f"meta_{split}.pkl")
        assert os.path.exists(meta_path), f"Meta file not found: {meta_path}"
        with open(meta_path, 'rb') as f:
            meta = pkl.load(f)
        metas += meta
        for idx, m in enumerate(meta):
            if 'uuid' not in m or 'prompt' not in m:
                raise KeyError(
                    f"Invalid DiffusionDB meta in {meta_path} at item {idx}, "
                    f"expected keys ['uuid', 'prompt'], got {list(m.keys())}"
                )
            meta_id = str(m['uuid'])
            prompt = m['prompt']
            id_to_prompt[meta_id] = prompt
    num_requests = len(metas)

    index = None
    id_map = []  # 存储样本ID顺序（DiffusionDB: uuid）
    latent_dim = None
    reuse_infos = {}  # sample_id -> (reuse_img_id, similarity, begin_timestep)
    total_start_time = 0
    total_time_cnt = 0

    for i in tqdm(range(num_requests), desc='Retrieving'):

        image_id = str(metas[i]['uuid'])
        prompt = metas[i]['prompt']

        # 查找时用CLIP编码prompt
        with torch.no_grad():
            text_tokens = clip.tokenize([prompt]).to(device)
            text_feat = clip_model.encode_text(text_tokens)
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
            text_feat_np = text_feat.cpu().numpy().astype(np.float32)

        if latent_dim is None:
            latent_dim = text_feat_np.shape[1]
            if DISTANCE_METRIC == 'cosine':
                index = faiss.IndexFlatIP(latent_dim)
            elif DISTANCE_METRIC == 'l2':
                index = faiss.IndexFlatL2(latent_dim)
            elif DISTANCE_METRIC == 'l1':
                index = faiss.IndexFlat(latent_dim, faiss.METRIC_L1)
            else:
                raise ValueError(f"Unsupported DISTANCE_METRIC: {DISTANCE_METRIC}")

        reuse_img_id = None
        similarity = None
        begin_timestep = None

        if index.ntotal > 0:
            start_time = time.time()
            D, I = index.search(text_feat_np, 1)
            sim = float(D[0][0])
            idx = int(I[0][0])
            end_time = time.time()
            total_start_time += (end_time - start_time)
            total_time_cnt += 1

            def similarity_to_reuse(sim):
                if sim > 0.0:
                    reuse_img_id = id_map[idx]
                    similarity = sim
                    begin_timestep = 15
                else:
                    reuse_img_id = None
                    similarity = None
                    begin_timestep = None
                return reuse_img_id, similarity, begin_timestep

            reuse_img_id, similarity, begin_timestep = similarity_to_reuse(sim)
            if reuse_img_id is not None and similarity is not None and begin_timestep is not None:
                print(f"[{i}/{num_requests}]: Found similar for image_id={image_id}: reuse_img_id={reuse_img_id}, similarity={similarity:.3f}")
            else:
                print(f"[{i}/{num_requests}]: No similar found for image_id={image_id}")
        else:
            print(f"[{i}/{num_requests}]: Index empty, no similar found for image_id={image_id}")

        reuse_infos[image_id] = (reuse_img_id, similarity, begin_timestep, prompt, id_to_prompt[reuse_img_id] if reuse_img_id is not None else None)

        # 存储时用CLIP编码原图
        org_path = f"{Config.TRAJ_ALL_SAVE_DIR}/{image_id}.png"
        assert os.path.exists(org_path), f"Original image `{org_path}` not found!"
        img_org = Image.open(org_path).convert('RGB')
        with torch.no_grad():
            image_input = clip_preprocess(img_org).unsqueeze(0).to(device)
            img_feat = clip_model.encode_image(image_input)
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
            img_feat_np = img_feat.cpu().numpy().astype(np.float32)

        index.add(img_feat_np)
        id_map.append(image_id)

    if total_time_cnt > 0:
        avg_time = total_start_time / total_time_cnt
    else:
        avg_time = 0.0
    print(f"Average retrieval time per query: {avg_time:.6f} seconds over {total_time_cnt} queries.")

    # 保存reuse_infos
    with open(REUSE_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(reuse_infos, f, ensure_ascii=True, indent=4)


