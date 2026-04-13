import importlib
import os

import clip
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3


TRUST_REMOTE_CODE_AESTHETICS = True
AESTHETICS_USE_FAST_PROCESSOR = True

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['USE_MODELSCOPE_HUB'] = '0'


def sorted_ids(id_set):
    return sorted(id_set, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))


def safe_float_stats(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr))


def summarize_named_lists(named_lists):
    rows = []
    for name, values in named_lists:
        mean_v, std_v = safe_float_stats(values)
        rows.append((name, mean_v, std_v))
    return rows


def print_metric_rows(metric_name, rows):
    print(f'\n===== {metric_name} =====')
    for name, mean_v, std_v in rows:
        if np.isnan(mean_v):
            print(f'{metric_name} | {name}: NaN')
        else:
            print(f'{metric_name} | {name}: {mean_v:.6f} +/- {std_v:.6f}')


def get_png_files_by_id(dir_path):
    def get_id(fname):
        return fname.split('_')[0] if '_' in fname else fname.split('.')[0]

    return {get_id(f): f for f in os.listdir(dir_path) if f.endswith('.png')}


def _import_transformers():
    try:
        return importlib.import_module('transformers')
    except ImportError:
        return None


def build_clip_runtime(device):
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
    clip_model.eval()
    return clip_model, clip_preprocess


def build_siglip2_runtime(device, ckpt_path, dtype):
    transformers = _import_transformers()
    if transformers is None:
        raise ImportError('transformers is required for SigLIP2 runtime.')

    siglip2_model = transformers.AutoModel.from_pretrained(ckpt_path, torch_dtype=dtype).to(device).eval()
    siglip2_processor = transformers.AutoProcessor.from_pretrained(ckpt_path)
    siglip2_max_text_len = int(siglip2_model.config.text_config.max_position_embeddings)
    return siglip2_model, siglip2_processor, siglip2_max_text_len


def compute_reconstruction_metrics(generated_img, reference_img, device):
    to_tensor = transforms.ToTensor()
    arr1 = to_tensor(generated_img).to(device)
    arr2 = to_tensor(reference_img).to(device)

    if arr1.shape != arr2.shape:
        arr2 = to_tensor(reference_img.resize(generated_img.size)).to(device)
    l2_dist = F.mse_loss(arr1, arr2).item()

    arr1_np = np.array(generated_img)
    arr2_np = np.array(reference_img)
    if arr1_np.shape != arr2_np.shape:
        arr2_np = np.array(reference_img.resize(generated_img.size))
    ssim_val = ssim(arr1_np, arr2_np, channel_axis=-1, data_range=255)
    psnr_val = psnr(arr1_np, arr2_np, data_range=255)

    return float(l2_dist), float(ssim_val), float(psnr_val)


def _get_clip_feat(image, clip_model, clip_preprocess, device):
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(image_input)
    return feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)


def compute_clip_metrics(generated_img, reference_img, prompt, clip_model, clip_preprocess, device):
    feat_img = _get_clip_feat(generated_img, clip_model, clip_preprocess, device)
    feat_img_org = _get_clip_feat(reference_img, clip_model, clip_preprocess, device)
    clip_sim = float((feat_img @ feat_img_org.T).item())

    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        feat_text = clip_model.encode_text(text)
        feat_text = feat_text / (feat_text.norm(dim=-1, keepdim=True) + 1e-8)
    clip_img_text = float((feat_img @ feat_text.T).item())
    clip_imgorg_text = float((feat_img_org @ feat_text.T).item())

    return clip_sim, clip_img_text, clip_imgorg_text


def compute_clip_image_text_single(image, prompt, clip_model, clip_preprocess, device):
    feat_img = _get_clip_feat(image, clip_model, clip_preprocess, device)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        feat_text = clip_model.encode_text(text)
        feat_text = feat_text / (feat_text.norm(dim=-1, keepdim=True) + 1e-8)
    return float((feat_img @ feat_text.T).item())


def compute_siglip2_single(image, prompt, model, processor, max_text_len, device):
    siglip2_inputs = processor(
        text=[prompt],
        images=[image],
        truncation=True,
        max_length=max_text_len,
        padding='max_length',
        return_tensors='pt',
    )
    siglip2_inputs = {k: v.to(device) for k, v in siglip2_inputs.items()}
    with torch.no_grad():
        siglip2_outputs = model(**siglip2_inputs)
    # SigLIP2 score is logits_per_image, not cosine dot-product.
    return float(siglip2_outputs.logits_per_image[0, 0].item())


def build_pickscore_runtime(device):
    transformers = _import_transformers()
    if transformers is None:
        print('Warning: transformers not installed, skip PickScore.')
        return None, None

    try:
        processor = transformers.AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        model = transformers.AutoModel.from_pretrained('yuvalkirstain/PickScore_v1').to(device)
        if not hasattr(model, 'get_image_features') or not hasattr(model, 'get_text_features'):
            raise TypeError(
                f'Loaded PickScore model type {type(model).__name__} does not expose '
                'get_image_features/get_text_features.'
            )
        model.eval()
        return model, processor
    except Exception as exc:
        print(f'Warning: failed to load PickScore model, skip PickScore. Details: {exc}')
        return None, None


def _pickscore_output_to_embedding(output, kind):
    # We need one embedding per sample with shape [B, D], not token-level [B, L, D].
    if torch.is_tensor(output):
        if output.ndim == 2:
            return output
        if output.ndim >= 3:
            return output[:, 0, :]

    if hasattr(output, 'image_embeds') and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, 'text_embeds') and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]

    if isinstance(output, dict):
        for key in ('image_embeds', 'text_embeds', 'pooler_output'):
            if key in output and output[key] is not None:
                return output[key]
        if 'last_hidden_state' in output and output['last_hidden_state'] is not None:
            return output['last_hidden_state'][:, 0, :]

    if isinstance(output, (tuple, list)) and len(output) > 0:
        # In current transformers versions for PickScore, get_*_features often returns
        # (last_hidden_state [B, L, C], pooler_output [B, D]). We must select [B, D].
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
        for item in output:
            if hasattr(item, 'pooler_output') or hasattr(item, 'image_embeds') or hasattr(item, 'text_embeds'):
                return _pickscore_output_to_embedding(item, kind)
        for item in output:
            if torch.is_tensor(item) and item.ndim >= 3:
                return item[:, 0, :]

    raise TypeError(f'Unsupported PickScore {kind} embedding output type: {type(output).__name__}')


def compute_pickscore_single(image, prompt, model, processor, device):
    if model is None or processor is None:
        return float('nan')

    try:
        with torch.no_grad():
            image_inputs = processor(images=[image], return_tensors='pt')
            text_inputs = processor(
                text=[prompt],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt',
            )
            image_inputs = image_inputs.to(device)
            text_inputs = text_inputs.to(device)

            # Prefer return_dict=False to get plain tensors on older/newer transformers variants.
            try:
                image_embs_raw = model.get_image_features(**image_inputs, return_dict=False)
            except TypeError:
                image_embs_raw = model.get_image_features(**image_inputs)

            try:
                text_embs_raw = model.get_text_features(**text_inputs, return_dict=False)
            except TypeError:
                text_embs_raw = model.get_text_features(**text_inputs)

            image_embs = F.normalize(_pickscore_output_to_embedding(image_embs_raw, 'image'), p=2, dim=-1)
            text_embs = F.normalize(_pickscore_output_to_embedding(text_embs_raw, 'text'), p=2, dim=-1)

            if image_embs.ndim != 2 or text_embs.ndim != 2:
                raise ValueError(
                    f'PickScore embeddings must be 2D [B, D], got image={tuple(image_embs.shape)}, '
                    f'text={tuple(text_embs.shape)}'
                )
            if image_embs.shape[-1] != text_embs.shape[-1]:
                raise ValueError(
                    f'PickScore embedding dim mismatch: image={tuple(image_embs.shape)}, '
                    f'text={tuple(text_embs.shape)}'
                )

            if hasattr(model, 'logit_scale'):
                # Single image/prompt score in the same form as official examples.
                score = (model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1))[0].item()
            else:
                score = (image_embs * text_embs).sum(dim=-1).item()
        return float(score)
    except Exception as exc:
        print(f'Warning: failed to compute PickScore for current sample. Details: {exc}')
        return float('nan')


def build_aesthetics_runtime(device):
    transformers = _import_transformers()
    if transformers is None:
        print('Warning: transformers not installed, skip Aesthetics Score.')
        return None, None

    os.environ['USE_MODELSCOPE_HUB'] = '0'
    model_id = 'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'

    try:
        processor = transformers.AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=TRUST_REMOTE_CODE_AESTHETICS,
            use_fast=AESTHETICS_USE_FAST_PROCESSOR,
        )
    except TypeError:
        processor = transformers.AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=TRUST_REMOTE_CODE_AESTHETICS,
        )

    load_errors = []
    config = None

    try:
        config = transformers.AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=TRUST_REMOTE_CODE_AESTHETICS,
        )
    except Exception as exc:
        load_errors.append(f'AutoConfig: {exc}')

    for loader_name, loader_cls in [
        ('AutoModelForImageClassification', transformers.AutoModelForImageClassification),
        ('AutoModel', transformers.AutoModel),
    ]:
        try:
            kwargs = {'trust_remote_code': TRUST_REMOTE_CODE_AESTHETICS}
            if config is not None:
                kwargs['config'] = config

            model = loader_cls.from_pretrained(model_id, **kwargs).to(device)
            model.eval()
            if loader_name != 'AutoModelForImageClassification':
                print(f'Info: loaded Aesthetics model with fallback loader {loader_name}.')
            return model, processor
        except Exception as exc:
            load_errors.append(f'{loader_name}: {exc}')

    if config is not None and TRUST_REMOTE_CODE_AESTHETICS:
        try:
            get_cls = importlib.import_module('transformers.dynamic_module_utils').get_class_from_dynamic_module
            architecture_names = list(getattr(config, 'architectures', []) or [])
            candidate_module_names = ['modeling_v2', 'modeling_predictor', 'modeling']
            class_candidates = architecture_names + [
                'AestheticsPredictorV2Linear',
                'AestheticsPredictor',
                'AestheticPredictor',
            ]

            tried_dynamic = []
            for module_name in candidate_module_names:
                for class_name in class_candidates:
                    spec = f'{module_name}.{class_name}'
                    try:
                        model_cls = get_cls(spec, model_id, trust_remote_code=True)
                        model = model_cls.from_pretrained(
                            model_id,
                            config=config,
                            trust_remote_code=True,
                        ).to(device)
                        model.eval()
                        print(f'Info: loaded Aesthetics model via dynamic class {spec}.')
                        return model, processor
                    except Exception as exc:
                        tried_dynamic.append(f'{spec}: {exc}')

            if tried_dynamic:
                load_errors.append(f"DynamicClass: {' | '.join(tried_dynamic)}")
        except Exception as exc:
            load_errors.append(f'DynamicClass bootstrap: {exc}')

    print(
        'Warning: failed to load Aesthetics model with all available loaders, skip Aesthetics Score. '
        f"Details: {' | '.join(load_errors)}. "
        'Try upgrading transformers (e.g., pip install -U transformers).'
    )
    return None, None


def _extract_aesthetics_scores(outputs):
    if torch.is_tensor(outputs):
        logits = outputs
    elif hasattr(outputs, 'logits'):
        logits = outputs.logits
    elif isinstance(outputs, dict):
        logits = None
        for key in ('logits', 'scores', 'score', 'predictions', 'prediction'):
            if key in outputs:
                logits = outputs[key]
                break
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        logits = outputs[0]
    else:
        logits = None

    if logits is None:
        raise ValueError('Aesthetics model output does not contain a recognizable score/logits field.')

    if isinstance(logits, (tuple, list)) and len(logits) > 0:
        logits = logits[0]
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)

    if logits.ndim == 0:
        logits = logits.unsqueeze(0)
    if logits.ndim > 1:
        logits = logits.squeeze(-1)

    return logits.detach().cpu().numpy().reshape(-1).tolist()


def compute_aesthetics_single(image, model, processor, device):
    if model is None or processor is None:
        return float('nan')

    try:
        with torch.no_grad():
            inputs = processor(images=[image], return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            values = _extract_aesthetics_scores(outputs)
            if len(values) == 0:
                return float('nan')
            return float(values[0])
    except Exception as exc:
        print(f'Warning: failed to compute Aesthetics score for current sample. Details: {exc}')
        return float('nan')


def compute_method_sample_metrics(method_name, gen_img_paths, org_img_paths, prompts, device, runtimes, enable):
    metrics = {}

    if enable.get('l2', False):
        metrics['l2'] = []
    if enable.get('ssim', False):
        metrics['ssim'] = []
    if enable.get('psnr', False):
        metrics['psnr'] = []
    if enable.get('clip_sim', False):
        metrics['clip_sim'] = []
    if enable.get('clip_img_text', False):
        metrics['clip_img_text'] = []

    if enable.get('siglip2', False):
        metrics['siglip2'] = []
    if enable.get('pickscore', False):
        metrics['pickscore'] = []
    if enable.get('aesthetics', False):
        metrics['aesthetics'] = []

    n = len(gen_img_paths)
    for idx in tqdm.trange(n, desc=f'{method_name} sample metrics'):
        gen_img = Image.open(gen_img_paths[idx]).convert('RGB')
        org_img = Image.open(org_img_paths[idx]).convert('RGB')
        prompt = prompts[idx]

        if enable.get('l2', False) or enable.get('ssim', False) or enable.get('psnr', False):
            l2_dist, ssim_val, psnr_val = compute_reconstruction_metrics(gen_img, org_img, device)
            if 'l2' in metrics:
                metrics['l2'].append(l2_dist)
            if 'ssim' in metrics:
                metrics['ssim'].append(ssim_val)
            if 'psnr' in metrics:
                metrics['psnr'].append(psnr_val)

        if enable.get('clip_sim', False) or enable.get('clip_img_text', False):
            clip_sim, clip_img_text, _ = compute_clip_metrics(
                gen_img,
                org_img,
                prompt,
                runtimes['clip_model'],
                runtimes['clip_preprocess'],
                device,
            )
            if 'clip_sim' in metrics:
                metrics['clip_sim'].append(clip_sim)
            if 'clip_img_text' in metrics:
                metrics['clip_img_text'].append(clip_img_text)

        if enable.get('siglip2', False):
            siglip2_score = compute_siglip2_single(
                gen_img,
                prompt,
                runtimes['siglip2_model'],
                runtimes['siglip2_processor'],
                runtimes['siglip2_max_text_len'],
                device,
            )
            metrics['siglip2'].append(siglip2_score)

        if enable.get('pickscore', False):
            pickscore = compute_pickscore_single(
                gen_img,
                prompt,
                runtimes['pickscore_model'],
                runtimes['pickscore_processor'],
                device,
            )
            metrics['pickscore'].append(pickscore)

        if enable.get('aesthetics', False):
            aes = compute_aesthetics_single(
                gen_img,
                runtimes['aesthetics_model'],
                runtimes['aesthetics_processor'],
                device,
            )
            metrics['aesthetics'].append(aes)

    return metrics


def compute_org_prompt_metrics(org_img_paths, prompts, device, runtimes, enable):
    metrics = {}
    if enable.get('clip_img_text', False):
        metrics['clip_img_text'] = []
    if enable.get('siglip2', False):
        metrics['siglip2'] = []
    if enable.get('pickscore', False):
        metrics['pickscore'] = []
    if enable.get('aesthetics', False):
        metrics['aesthetics'] = []

    n = len(org_img_paths)
    for idx in tqdm.trange(n, desc='Org sample metrics'):
        img = Image.open(org_img_paths[idx]).convert('RGB')
        prompt = prompts[idx]

        if enable.get('clip_img_text', False):
            metrics['clip_img_text'].append(
                compute_clip_image_text_single(
                    img,
                    prompt,
                    runtimes['clip_model'],
                    runtimes['clip_preprocess'],
                    device,
                )
            )

        if enable.get('siglip2', False):
            metrics['siglip2'].append(
                compute_siglip2_single(
                    img,
                    prompt,
                    runtimes['siglip2_model'],
                    runtimes['siglip2_processor'],
                    runtimes['siglip2_max_text_len'],
                    device,
                )
            )

        if enable.get('pickscore', False):
            metrics['pickscore'].append(
                compute_pickscore_single(
                    img,
                    prompt,
                    runtimes['pickscore_model'],
                    runtimes['pickscore_processor'],
                    device,
                )
            )

        if enable.get('aesthetics', False):
            metrics['aesthetics'].append(
                compute_aesthetics_single(
                    img,
                    runtimes['aesthetics_model'],
                    runtimes['aesthetics_processor'],
                    device,
                )
            )

    return metrics


def build_inception_runtime():
    if not torch.cuda.is_available():
        print('Warning: CUDA is not available. Inception-based metrics may be slow on CPU.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights, transform_input=False).to(device)
    inception.eval()
    preprocess_fid = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    preprocess_is = weights.transforms()
    return inception, preprocess_fid, preprocess_is, device


def extract_inception_features(img_paths, dir_name, inception, preprocess, device, batch_size=100):
    feats = []
    n = len(img_paths)
    with torch.no_grad():
        for start in tqdm.trange(0, n, batch_size, desc=f'Extracting {dir_name} features'):
            end = min(start + batch_size, n)
            batch_paths = img_paths[start:end]
            batch_imgs = [Image.open(p).convert('RGB') for p in batch_paths]
            batch_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
            del batch_imgs
            features = inception.Conv2d_1a_3x3(batch_tensors)
            features = inception.Conv2d_2a_3x3(features)
            features = inception.Conv2d_2b_3x3(features)
            features = torch.nn.functional.max_pool2d(features, kernel_size=3, stride=2)
            features = inception.Conv2d_3b_1x1(features)
            features = inception.Conv2d_4a_3x3(features)
            features = torch.nn.functional.max_pool2d(features, kernel_size=3, stride=2)
            features = inception.Mixed_5b(features)
            features = inception.Mixed_5c(features)
            features = inception.Mixed_5d(features)
            features = inception.Mixed_6a(features)
            features = inception.Mixed_6b(features)
            features = inception.Mixed_6c(features)
            features = inception.Mixed_6d(features)
            features = inception.Mixed_6e(features)
            features = inception.Mixed_7a(features)
            features = inception.Mixed_7b(features)
            features = inception.Mixed_7c(features)
            batch_feats = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1).cpu().numpy()
            if batch_feats.ndim == 1:
                batch_feats = batch_feats[None, :]
            feats.append(batch_feats)
    return np.concatenate(feats, axis=0)


def extract_inception_probs(img_paths, dir_name, inception, preprocess, device, batch_size=64):
    probs_all = []
    n = len(img_paths)
    with torch.no_grad():
        for start in tqdm.trange(0, n, batch_size, desc=f'Extracting {dir_name} probs'):
            end = min(start + batch_size, n)
            batch_paths = img_paths[start:end]
            batch_imgs = [Image.open(p).convert('RGB') for p in batch_paths]
            batch_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
            del batch_imgs
            logits = inception(batch_tensors)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)


def compute_inception_score_from_probs(probs, splits=10):
    n = probs.shape[0]
    if n < 2:
        raise ValueError('Need at least 2 samples to compute Inception Score.')
    if splits <= 0:
        raise ValueError('splits must be positive.')

    splits = min(splits, n)
    scores = []
    eps = 1e-16

    for k in range(splits):
        part = probs[k * n // splits:(k + 1) * n // splits]
        if part.shape[0] == 0:
            continue
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        split_is = float(np.exp(np.mean(np.sum(kl, axis=1))))
        scores.append(split_is)

    if len(scores) == 0:
        raise ValueError('No valid splits for Inception Score.')

    return float(np.mean(scores)), float(np.std(scores))


def compute_fid_from_feats(feats1, feats2):
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    diff = mu1 - mu2
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


def _polynomial_mmd2_unbiased(x, y, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    k_xx = (gamma * (x @ x.T) + coef0) ** degree
    k_yy = (gamma * (y @ y.T) + coef0) ** degree
    k_xy = (gamma * (x @ y.T) + coef0) ** degree

    m = x.shape[0]
    n = y.shape[0]
    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    sum_xy = k_xy.sum() / (m * n)
    return sum_xx + sum_yy - 2.0 * sum_xy


def compute_kid_from_feats(feats1, feats2):
    n1 = feats1.shape[0]
    n2 = feats2.shape[0]
    if min(n1, n2) < 2:
        raise ValueError('Need at least 2 samples in each set to compute KID.')
    if n1 > n2:
        feats1 = feats1[:n2]
    elif n2 > n1:
        feats2 = feats2[:n1]
    kid_score = _polynomial_mmd2_unbiased(feats1, feats2)
    return float(kid_score), 0.0


def export_metrics_to_excel(summary_rows_by_metric, output_excel_path):
    try:
        pd = importlib.import_module('pandas')
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            for metric_name, rows in summary_rows_by_metric.items():
                df = pd.DataFrame(rows, columns=['Method', 'Mean', 'Std'])
                df.to_excel(writer, sheet_name=metric_name, index=False)
    except ImportError:
        try:
            Workbook = importlib.import_module('openpyxl').Workbook
        except ImportError as exc:
            raise ImportError(
                'Excel export requires pandas or openpyxl. Install one of them: pip install pandas openpyxl'
            ) from exc

        wb = Workbook()
        default_sheet = wb.active
        wb.remove(default_sheet)

        for metric_name, rows in summary_rows_by_metric.items():
            ws = wb.create_sheet(title=metric_name)
            ws.append(['Method', 'Mean', 'Std'])
            for method, mean_value, std_value in rows:
                ws.append([method, mean_value, std_value])

        wb.save(output_excel_path)

    print(f'\nSaved metric means/std to Excel: {output_excel_path}')
