import os
import json
import math

import torch
from config import Config

from load_dataset import load_dataset
from metric_runtime import (
    build_aesthetics_runtime,
    build_clip_runtime,
    build_inception_runtime,
    build_pickscore_runtime,
    build_siglip2_runtime,
    compute_fid_from_feats,
    compute_inception_score_from_probs,
    compute_kid_from_feats,
    compute_method_sample_metrics,
    compute_org_prompt_metrics,
    export_metrics_to_excel,
    extract_inception_features,
    extract_inception_probs,
    get_png_files_by_id,
    print_metric_rows,
    sorted_ids,
    summarize_named_lists,
)


# =========================
# Metric Switches (Global)
# =========================
ENABLE_SSIM = True
ENABLE_PSNR = True
ENABLE_L2 = True
ENABLE_CLIP_SIM = True
ENABLE_CLIP_IMG_TEXT = True
ENABLE_SIGLIP2 = True
ENABLE_PICKSCORE = True
ENABLE_AESTHETICS = False

ENABLE_FID = True
ENABLE_KID = False
ENABLE_IS = True

SIGLIP2_CKPT = 'model_ckpt/siglip2-giant-opt-patch16-384'
SIGLIP2_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

ORG_DIR = Config.TRAJ_ALL_SAVE_DIR
SAVE_DIR = Config.COMPARE_SAVE_DIR
os.makedirs(SAVE_DIR, exist_ok=True)
EXCEL_PATH = os.path.join(SAVE_DIR, 'metrics_summary.xlsx')
JSON_PATH = os.path.join(SAVE_DIR, 'comparison_results.json')

METHOD_DIRS = Config.COMPARE_METHOD_DIRS


def build_prompt_map():
    df = load_dataset()
    if 'uuid' not in df.columns or 'prompt' not in df.columns:
        raise ValueError("load_dataset() must return columns 'uuid' and 'prompt'.")
    prompt_map = {str(uuid): str(prompt) for uuid, prompt in zip(df['uuid'], df['prompt'])}
    if len(prompt_map) == 0:
        raise ValueError('Prompt mapping is empty. Please check dataset/cache files.')
    return prompt_map


def _to_json_float(value):
    v = float(value)
    if not math.isfinite(v):
        return None
    return v


def _rows_to_summary_dict(rows):
    summary_dict = {}
    for method, mean_v, std_v in rows:
        summary_dict[method] = {
            'mean': _to_json_float(mean_v),
            'std': _to_json_float(std_v),
        }
    return summary_dict


def _build_per_image_metric(common_ids, method_dirs, method_metrics, metric_key):
    per_image = {}
    for method_name, _ in method_dirs:
        values = method_metrics[method_name][metric_key]
        if len(values) != len(common_ids):
            raise ValueError(
                f'Length mismatch for metric "{metric_key}" in method "{method_name}": '
                f'len(values)={len(values)} vs len(common_ids)={len(common_ids)}'
            )
        per_image[method_name] = {
            str(sample_id): _to_json_float(values[idx]) for idx, sample_id in enumerate(common_ids)
        }
    return per_image


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    prompt_map = build_prompt_map()

    org_png_by_id = get_png_files_by_id(ORG_DIR)
    method_png_by_id = {name: get_png_files_by_id(dir_path) for name, dir_path in METHOD_DIRS}

    common_ids = set(org_png_by_id.keys())
    for name, _ in METHOD_DIRS:
        common_ids &= set(method_png_by_id[name].keys())
    common_ids = sorted_ids(common_ids)

    if len(common_ids) == 0:
        raise ValueError(f'No common ids across {ORG_DIR} and all configured generated directories.')

    need_prompt = ENABLE_CLIP_IMG_TEXT or ENABLE_SIGLIP2 or ENABLE_PICKSCORE
    if need_prompt:
        missing_prompt_ids = [sample_id for sample_id in common_ids if sample_id not in prompt_map]
        if len(missing_prompt_ids) > 0:
            raise ValueError(
                f'Missing prompts for {len(missing_prompt_ids)} ids (e.g., {missing_prompt_ids[:5]}). '
                'Please ensure load_dataset covers all generated uuids.'
            )

    prompts = [prompt_map.get(sample_id, '') for sample_id in common_ids]
    org_img_paths = [os.path.join(ORG_DIR, org_png_by_id[sample_id]) for sample_id in common_ids]

    enable = {
        'l2': ENABLE_L2,
        'ssim': ENABLE_SSIM,
        'psnr': ENABLE_PSNR,
        'clip_sim': ENABLE_CLIP_SIM,
        'clip_img_text': ENABLE_CLIP_IMG_TEXT,
        'siglip2': ENABLE_SIGLIP2,
        'pickscore': ENABLE_PICKSCORE,
        'aesthetics': ENABLE_AESTHETICS,
    }

    runtimes = {}
    if ENABLE_CLIP_SIM or ENABLE_CLIP_IMG_TEXT:
        clip_model, clip_preprocess = build_clip_runtime(device)
        runtimes['clip_model'] = clip_model
        runtimes['clip_preprocess'] = clip_preprocess

    if ENABLE_SIGLIP2:
        siglip2_model, siglip2_processor, siglip2_max_text_len = build_siglip2_runtime(device, SIGLIP2_CKPT, SIGLIP2_DTYPE)
        runtimes['siglip2_model'] = siglip2_model
        runtimes['siglip2_processor'] = siglip2_processor
        runtimes['siglip2_max_text_len'] = siglip2_max_text_len

    if ENABLE_PICKSCORE:
        pickscore_model, pickscore_processor = build_pickscore_runtime(device)
        runtimes['pickscore_model'] = pickscore_model
        runtimes['pickscore_processor'] = pickscore_processor

    if ENABLE_AESTHETICS:
        aesthetics_model, aesthetics_processor = build_aesthetics_runtime(device)
        runtimes['aesthetics_model'] = aesthetics_model
        runtimes['aesthetics_processor'] = aesthetics_processor

    method_metrics = {}
    for method_name, dir_path in METHOD_DIRS:
        gen_img_paths = [os.path.join(dir_path, method_png_by_id[method_name][sample_id]) for sample_id in common_ids]
        method_metrics[method_name] = compute_method_sample_metrics(
            method_name=method_name,
            gen_img_paths=gen_img_paths,
            org_img_paths=org_img_paths,
            prompts=prompts,
            device=device,
            runtimes=runtimes,
            enable=enable,
        )

    summary = {}
    comparison_results = {}

    sample_metric_specs = [
        ('SSIM', 'ssim', ENABLE_SSIM),
        ('PSNR', 'psnr', ENABLE_PSNR),
        ('L2', 'l2', ENABLE_L2),
        ('CLIP_IMG_IMG', 'clip_sim', ENABLE_CLIP_SIM),
        ('CLIP_IMG_TEXT', 'clip_img_text', ENABLE_CLIP_IMG_TEXT),
        ('SigLIP2', 'siglip2', ENABLE_SIGLIP2),
        ('PickScore', 'pickscore', ENABLE_PICKSCORE),
        ('Aesthetics', 'aesthetics', ENABLE_AESTHETICS),
    ]
    for metric_name, metric_key, enabled_flag in sample_metric_specs:
        if enabled_flag:
            comparison_results[metric_name] = {
                'per_image': _build_per_image_metric(common_ids, METHOD_DIRS, method_metrics, metric_key)
            }

    if ENABLE_SSIM:
        rows = summarize_named_lists([(name, method_metrics[name]['ssim']) for name, _ in METHOD_DIRS])
        print_metric_rows('SSIM', rows)
        summary['SSIM'] = rows
        comparison_results['SSIM']['summary'] = _rows_to_summary_dict(rows)

    if ENABLE_PSNR:
        rows = summarize_named_lists([(name, method_metrics[name]['psnr']) for name, _ in METHOD_DIRS])
        print_metric_rows('PSNR', rows)
        summary['PSNR'] = rows
        comparison_results['PSNR']['summary'] = _rows_to_summary_dict(rows)

    if ENABLE_L2:
        rows = summarize_named_lists([(name, method_metrics[name]['l2']) for name, _ in METHOD_DIRS])
        print_metric_rows('L2', rows)
        summary['L2'] = rows
        comparison_results['L2']['summary'] = _rows_to_summary_dict(rows)

    if ENABLE_CLIP_SIM:
        rows = summarize_named_lists([(name, method_metrics[name]['clip_sim']) for name, _ in METHOD_DIRS])
        print_metric_rows('CLIP-Image-Image', rows)
        summary['CLIP_IMG_IMG'] = rows
        comparison_results['CLIP_IMG_IMG']['summary'] = _rows_to_summary_dict(rows)

    if ENABLE_CLIP_IMG_TEXT:
        rows = summarize_named_lists([(name, method_metrics[name]['clip_img_text']) for name, _ in METHOD_DIRS])
        print_metric_rows('CLIP-Image-Text', rows)
        summary['CLIP_IMG_TEXT'] = rows
        comparison_results['CLIP_IMG_TEXT']['summary'] = _rows_to_summary_dict(rows)

    if ENABLE_SIGLIP2 or ENABLE_PICKSCORE or ENABLE_AESTHETICS or ENABLE_CLIP_IMG_TEXT:
        org_png_ids = sorted_ids(set(org_png_by_id.keys()))
        org_prompt_ids = [sample_id for sample_id in org_png_ids if sample_id in prompt_map]
        org_prompts = [prompt_map[sample_id] for sample_id in org_prompt_ids]
        org_paths_for_text_metrics = [os.path.join(ORG_DIR, org_png_by_id[sample_id]) for sample_id in org_prompt_ids]
        print(f'Using {len(org_paths_for_text_metrics)} org images for prompt-based/org-only metrics.')

        org_metrics = compute_org_prompt_metrics(
            org_img_paths=org_paths_for_text_metrics,
            prompts=org_prompts,
            device=device,
            runtimes=runtimes,
            enable=enable,
        )
    else:
        org_metrics = {}

    if ENABLE_SIGLIP2:
        rows = summarize_named_lists(
            [(name, method_metrics[name]['siglip2']) for name, _ in METHOD_DIRS] + [('Org', org_metrics['siglip2'])]
        )
        print_metric_rows('SigLIP2-Logits', rows)
        summary['SigLIP2'] = rows
        comparison_results['SigLIP2']['summary'] = _rows_to_summary_dict(rows)

    if ENABLE_PICKSCORE:
        rows = summarize_named_lists(
            [(name, method_metrics[name]['pickscore']) for name, _ in METHOD_DIRS] + [('Org', org_metrics['pickscore'])]
        )
        print_metric_rows('PickScore', rows)
        summary['PickScore'] = rows
        comparison_results['PickScore']['summary'] = _rows_to_summary_dict(rows)

    if ENABLE_AESTHETICS:
        rows = summarize_named_lists(
            [(name, method_metrics[name]['aesthetics']) for name, _ in METHOD_DIRS] + [('Org', org_metrics['aesthetics'])]
        )
        print_metric_rows('Aesthetics', rows)
        summary['Aesthetics'] = rows
        comparison_results['Aesthetics']['summary'] = _rows_to_summary_dict(rows)

    if ENABLE_CLIP_IMG_TEXT:
        rows = summarize_named_lists(
            [(name, method_metrics[name]['clip_img_text']) for name, _ in METHOD_DIRS] + [('Org', org_metrics['clip_img_text'])]
        )
        print_metric_rows('CLIP-Image-Text-With-Org', rows)

    if ENABLE_FID or ENABLE_KID or ENABLE_IS:
        inception, preprocess_fid, preprocess_is, inception_device = build_inception_runtime()
    else:
        inception = None
        preprocess_fid = None
        preprocess_is = None
        inception_device = None

    if ENABLE_FID or ENABLE_KID:
        print('\n===== FID/KID (configured methods only) =====')
        org_feats = extract_inception_features(org_img_paths, ORG_DIR, inception, preprocess_fid, inception_device)

        if ENABLE_FID:
            fid_rows = []
        if ENABLE_KID:
            kid_rows = []

        for method_name, dir_path in METHOD_DIRS:
            gen_img_paths = [os.path.join(dir_path, method_png_by_id[method_name][sample_id]) for sample_id in common_ids]
            gen_feats = extract_inception_features(gen_img_paths, dir_path, inception, preprocess_fid, inception_device)
            if ENABLE_FID:
                fid_rows.append((method_name, compute_fid_from_feats(gen_feats, org_feats), 0.0))
            if ENABLE_KID:
                kid_mean, kid_std = compute_kid_from_feats(gen_feats, org_feats)
                kid_rows.append((method_name, kid_mean, kid_std))

        if ENABLE_FID:
            print_metric_rows('FID', fid_rows)
            summary['FID'] = fid_rows
            comparison_results['FID'] = {'summary': _rows_to_summary_dict(fid_rows)}
        if ENABLE_KID:
            print_metric_rows('KID', kid_rows)
            summary['KID'] = kid_rows
            comparison_results['KID'] = {'summary': _rows_to_summary_dict(kid_rows)}

    if ENABLE_IS:
        print('\n===== IS (configured methods only) =====')
        is_rows = []
        for method_name, dir_path in METHOD_DIRS:
            gen_img_paths = [os.path.join(dir_path, method_png_by_id[method_name][sample_id]) for sample_id in common_ids]
            probs = extract_inception_probs(gen_img_paths, dir_path, inception, preprocess_is, inception_device)
            is_mean, is_std = compute_inception_score_from_probs(probs, splits=10)
            is_rows.append((method_name, is_mean, is_std))
        print_metric_rows('IS', is_rows)
        summary['IS'] = is_rows
        comparison_results['IS'] = {'summary': _rows_to_summary_dict(is_rows)}

    export_metrics_to_excel(summary, EXCEL_PATH)

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    print(f'Saved detailed metric results to JSON: {JSON_PATH}')


if __name__ == '__main__':
    main()
