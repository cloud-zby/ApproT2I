import json
import os
import pandas as pd
from config import Config


def _normalize_uuid_prompt_df(df):
    if 'uuid' in df.columns and 'prompt' in df.columns:
        out = df[['uuid', 'prompt']].copy()
    elif 'best_image_uid' in df.columns and 'caption' in df.columns:
        out = df[['best_image_uid', 'caption']].rename(
            columns={'best_image_uid': 'uuid', 'caption': 'prompt'}
        )
    elif 'image_id' in df.columns and 'caption' in df.columns:
        out = df[['image_id', 'caption']].rename(columns={'image_id': 'uuid', 'caption': 'prompt'})
    elif 'image_name' in df.columns and 'prompt' in df.columns:
        out = df[['image_name', 'prompt']].copy()
        out['image_name'] = out['image_name'].astype(str)
        out['uuid'] = out['image_name'].str.replace('.png', '', regex=False)
        out = out[['uuid', 'prompt']]
    else:
        raise ValueError(
            'Cannot parse uuid/prompt columns from dataframe. '\
            'Expected one of: (uuid,prompt), (best_image_uid,caption), (image_id,caption), (image_name,prompt).'
        )

    out['uuid'] = out['uuid'].astype(str)
    out['prompt'] = out['prompt'].astype(str)
    out = out.dropna(subset=['uuid', 'prompt'])
    out = out.drop_duplicates(subset=['uuid'], keep='first').reset_index(drop=True)
    return out


def load_coco_dataset(json_path='dataset/COCO2014_dataset.json'):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'COCO dataset file not found: {json_path}')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'annotations' not in data:
        raise ValueError('COCO json format error: missing top-level key "annotations".')

    annotations = data['annotations']
    if not isinstance(annotations, list):
        raise ValueError('COCO json format error: "annotations" must be a list.')

    # Follow prior project logic: for each image_id keep the first caption.
    first_caption_by_image = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get('image_id')
        caption = ann.get('caption')
        if image_id is None or caption is None:
            continue
        image_id = str(image_id)
        if image_id not in first_caption_by_image:
            first_caption_by_image[image_id] = str(caption)

    df = pd.DataFrame(
        {'uuid': list(first_caption_by_image.keys()), 'prompt': list(first_caption_by_image.values())}
    )
    df = _normalize_uuid_prompt_df(df)
    print(f'Loaded COCO dataset from {json_path}, total records: {len(df)}')
    return df


def load_diffusiondb_dataset(parquet_path='dataset/DiffusionDB_dataset.parquet'):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f'DiffusionDB dataset file not found: {parquet_path}')

    df = pd.read_parquet(parquet_path)
    df = _normalize_uuid_prompt_df(df)
    print(f'Loaded DiffusionDB dataset from {parquet_path}, total records: {len(df)}')
    return df


def load_pickapic_dataset(parquet_path='dataset/PicaPic_dataset.parquet'):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f'Pick-a-Pic dataset file not found: {parquet_path}')

    df = pd.read_parquet(parquet_path)
    df = _normalize_uuid_prompt_df(df)
    print(f'Loaded Pick-a-Pic dataset from {parquet_path}, total records: {len(df)}')
    return df


def load_dataset(
    dataset_name=Config.DATASET_NAME,
    coco_json_path='dataset/COCO2014_dataset.json',
    diffusiondb_parquet_path='dataset/DiffusionDB_dataset.parquet',
    pickapic_parquet_path='dataset/PicaPic_dataset.parquet',
):
    dataset_key = str(dataset_name).strip().lower().replace('_', '-').replace(' ', '-')

    if dataset_key == 'coco':
        return load_coco_dataset(coco_json_path)

    if dataset_key in ('diffusiondb', 'diffusion-db'):
        return load_diffusiondb_dataset(diffusiondb_parquet_path)

    if dataset_key in ('pick-a-pic', 'pickapic', 'picka-pic', 'pick-apic'):
        return load_pickapic_dataset(pickapic_parquet_path)

    raise ValueError(
        f'Unsupported dataset_name: {dataset_name}. '\
        'Supported values: coco, diffusiondb, pick-a-pic.'
    )