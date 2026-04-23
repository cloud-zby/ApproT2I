from tqdm import tqdm
import torch
import os
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
TOP_K = Config.TOP_K
REUSE_INFO_BY_PROMPT_PATH = Config.REUSE_INFO_BY_PROMPT_PATH

if os.path.exists(REUSE_INFO_BY_PROMPT_PATH):
    print(f"Warning: {REUSE_INFO_BY_PROMPT_PATH} already exists and will be skipped. Please delete it if you want to re-run retrieval.")
else:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 加载CLIP模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, _ = clip.load("ViT-B/32", device=device)
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
            sample_id = str(m['uuid'])
            id_to_prompt[sample_id] = m['prompt']
    num_requests = len(metas)

    index = None
    id_map = []  # 存储样本ID顺序（DiffusionDB: uuid）
    latent_dim = None
    reuse_infos = {}  # sample_id -> query results
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

        top_matches = []

        if index.ntotal > 0:
            start_time = time.time()
            k = min(TOP_K, index.ntotal)
            D, I = index.search(text_feat_np, k)
            end_time = time.time()
            total_start_time += (end_time - start_time)
            total_time_cnt += 1

            for rank, (score, faiss_idx) in enumerate(zip(D[0], I[0]), start=1):
                if faiss_idx < 0:
                    continue

                candidate_id = id_map[int(faiss_idx)]
                candidate_prompt = id_to_prompt.get(candidate_id)

                # For L2/L1, smaller distance means more similar. Convert to a larger-is-better similarity.
                similarity = float(score) if DISTANCE_METRIC == 'cosine' else float(-score)

                top_matches.append({
                    'rank': rank,
                    'reuse_img_id': candidate_id,
                    'similarity': similarity,
                    'candidate_prompt': candidate_prompt,
                })

            if len(top_matches) > 0:
                best = top_matches[0]
                print(
                    f"[{i}/{num_requests}]: Found top-{len(top_matches)} for uuid={image_id}, "
                    f"best reuse_img_id={best['reuse_img_id']}, similarity={best['similarity']:.3f}"
                )
            else:
                print(f"[{i}/{num_requests}]: No similar found for uuid={image_id}")
        else:
            print(f"[{i}/{num_requests}]: Index empty, no similar found for uuid={image_id}")

        reuse_infos[image_id] = {
            'query_prompt': prompt,
            'top_k': top_matches,
        }

        # 存储时也使用请求prompt的文本embedding，实现文本-文本检索。
        index.add(text_feat_np)
        id_map.append(image_id)

    if total_time_cnt > 0:
        print(f"Average retrieval time per query: {total_start_time / total_time_cnt:.6f} seconds over {total_time_cnt} queries.")
    else:
        print("Average retrieval time per query: N/A (no retrieval executed).")

    # 保存reuse_infos
    with open(REUSE_INFO_BY_PROMPT_PATH, "w", encoding="utf-8") as f:
        json.dump(reuse_infos, f, ensure_ascii=True, indent=4)


