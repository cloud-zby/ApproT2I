import torch


class Config:
    # Dataset file paths
    COCO_JSON_PATH = 'dataset/COCO2014_dataset.json'
    DIFFUSIONDB_PARQUET_PATH = 'dataset/DiffusionDB_dataset.parquet'
    PICKAPIC_PARQUET_PATH = 'dataset/PicaPic_dataset.parquet'

    # Dataset selector used by load_dataset().
    DATASET_NAME = 'diffusiondb'  # 'coco' or 'diffusiondb' or 'pick-a-pic'

    # Pipeline class selector: 'SD2.1Pipeline' or 'SDXLPipeline' or 'FLUX2KleinPipeline'.
    PIPELINE_CLASS_NAME = 'SDXLPipeline'

    # Model checkpoint paths for supported pipeline classes.
    MODEL_CKPT_PATHS = {
        'SDXLPipeline': './model_ckpt/stable-diffusion-xl-base-1.0',
        'SD2.1Pipeline': './model_ckpt/stable-diffusion-2-1',
        'FLUX2KleinPipeline': './model_ckpt/FLUX.2-klein-base-9B',
    }

    # Unified model checkpoint path resolved from PIPELINE_CLASS_NAME.
    MODEL_CKPT_PATH = MODEL_CKPT_PATHS[PIPELINE_CLASS_NAME]

    # Seed for shuffling dataset indices before split processing.
    DATA_SHUFFLE_SEED = 19260817

    # Number of denoising steps for full trajectory generation.
    TRAJ_ALL_TOTAL_INFERENCE_STEPS = 50

    # Output dir for split metadata dumped by traj_all.py.
    TRAJ_ALL_META_PKL_PATH_TEMPLATE = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/"

    # Seed used in retrieval pipelines.
    RETRIEVE_SEED = 42

    # Retrieval metric for FAISS search: cosine, l2, or l1.
    DISTANCE_METRIC = 'cosine'.lower()

    # Top-k candidates used in prompt-based retrieval and recon fusion.
    TOP_K = 3

    # Weight for similarity-based normalization in Recon method. The final weight is computed as: norm_weight = RECON_NORM_WEIGHT * (similarity / max_similarity) + (1 - RECON_NORM_WEIGHT). Suggested range: 0.4 to 0.6, where higher values give more emphasis to similarity differences and lower values lead to more uniform weights among candidates.
    RECON_NORM_WEIGHT = 0.4

    # Noise scale added to the weighted latent in Recon/ReDi/NIRVANA method to enhance diversity in ReCon. Suggested range: 0.0 (no noise) to 0.2 (strong noise).
    ADD_NOISE_SCALE = 0.10

    # Whether to perform global latent normalization in MoDM.
    MoDM_DO_NORM = True

    # Reuse-info JSON path for image/prompt retrieval output and consumers.
    REUSE_INFO_PATH = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/reuse_infos_{DISTANCE_METRIC}.json"

    # Reuse-info JSON path for prompt-only top-k retrieval output and consumers.
    REUSE_INFO_BY_PROMPT_PATH = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/reuse_infos_{DISTANCE_METRIC}_by_prompt.json"

    # Number of denoising steps for all reuse-based generation methods.
    GENERATION_TOTAL_INFERENCE_STEPS = 50

    # Start timestep for reuse-based generation methods.
    GENERATION_BEGIN_TIMESTEP = 15

    # Timesteps for capturing small-step trajectories and attention maps. These should correspond to the timesteps used in attention_small_steps_*.py scripts. Suggested to cover the early denoising steps where retrieval-based intervention is most effective.
    """
    SD2.1
    [981, 961, 941, 921, 901, 
    881, 861, 841, 821, 801, 
    781, 761, 741, 721, 701, 
    681, 661, 641, 621, 601, 
    581, 561, 541, 521, 501, 
    481, 461, 441, 421, 401, 
    381, 361, 341, 321, 301, 
    281, 261, 241, 221, 201, 
    181, 161, 141, 121, 101,  
    81, 61, 41, 21, 1]
    SDXL
    [981, 961, 941, 921, 901, 
    881, 861, 841, 821, 801, 
    781, 761, 741, 721, 701, 
    681, 661, 641, 621, 601, 
    581, 561, 541, 521, 501, 
    481, 461, 441, 421, 401, 
    381, 361, 341, 321, 301, 
    281, 261, 241, 221, 201, 
    181, 161, 141, 121, 101,  
    81, 61, 41, 21, 1]
    FLUX
    [1000.0000,  997.3091,  994.5214,  991.6316,  988.6340,  
    985.5225,  982.2905,  978.9307,  975.4355,  971.7964,  
    968.0046,  964.0501,  959.9222,  955.6093,  951.0987,  
    946.3763,  941.4270,  936.2340,  930.7789,  925.0412,  
    918.9985,  912.6257,  905.8951,  898.7756,  891.2326,  
    883.2271,  874.7151,  865.6472,  855.9669,  845.6102,
    834.5034,  822.5620,  809.6884,  795.7689,  780.6708,  
    764.2374,  746.2838,  726.5886,  704.8854,  680.8509,  
    654.0879,  624.1038,  590.2800,  551.8284,  507.7296,  
    456.6408,  396.7571,  325.5933,  239.6312,  133.7189]
    """
    TIMESTEPS_FOR_SMALL_STEPS_DICT = {
        'SD2.1Pipeline': [981, 901, 701],
        'SDXLPipeline': [901, 801, 701, 601, 501],
        'FLUX2KleinPipeline': [1000.0000,  988.4957,  974.4825,  957.0386,  934.7291, 905.1879, 864.2187,  803.6000,  704.7356,  514.7510],
    }
    TIMESTEPS_FOR_SMALL_STEPS = TIMESTEPS_FOR_SMALL_STEPS_DICT[PIPELINE_CLASS_NAME]

    # Number of denoising steps for small-step trajectory generation in FLUX. It shoud be according to TIMESTEPS_FOR_SMALL_STEPS_DICT for FLUX2KleinPipeline. For SD2.1 and SDXL pipelines, we don't need to set.
    SMALL_STEPS_INFERENCE_STEPS_FLUX = 10

    # Number of self trajectory steps used before latent fusion.
    PRELIMINARY_STEPS = 3

    # Weight for self trajectory in Ours method. The final latent is a weighted sum of self and retrieved latents, where this weight controls the contribution of the self trajectory.
    OURS_SELF_WEIGHT = 0.5

    # Weight for self trajectory in Ours method. The final latent is a weighted sum of self and retrieved latents, where this weight controls the contribution of the self trajectory. Suggested range: 0.46 to 0.68.
    OURS_MIX_SELF_WEIGHT_MIN = 0.4
    OURS_MIX_SELF_WEIGHT_MAX = 0.65

    # Seed for torch.Generator in every pipeline sampling call.
    GENERATOR_SEED = 1024

    # Classifier-free guidance scale used in every generation call.
    GUIDANCE_SCALES = {
        'SD2.1Pipeline': 7.5,
        'SDXLPipeline': 7.5,
        'FLUX2KleinPipeline': 4.0,
    }
    GUIDANCE_SCALE = GUIDANCE_SCALES[PIPELINE_CLASS_NAME]

    # Directory for full-trajectory images and latent trajectories.
    TRAJ_ALL_SAVE_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/org_images"

    # Directory for short-step images and latent trajectories.
    SMALL_STEPS_IMAGE_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/small_steps_images"

    # Directory for short-step attention matrices.
    SMALL_STEPS_ATTN_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/small_steps_attn"

    # Output directory for ReDi generated images.
    REDI_OUTPUT_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/redi_generated"

    # Output directory for MoDM generated images.
    MODM_OUTPUT_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/modm_generated"

    # Output directory for Ours generated images.
    OURS_OUTPUT_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/ours_generated"

    # Output directory for Ours(global norm) generated images.
    OURS_GLOBAL_NORM_OUTPUT_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/ours_generated_global_norm"

    # Output directory for OursMix generated images.
    OURS_MIX_OUTPUT_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/ours_generated_mix"

    # Output directory for OursMix(global norm) generated images.
    OURS_MIX_GLOBAL_NORM_OUTPUT_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/ours_generated_mix_global_norm"

    # Output directory for Recon generated images.
    RECON_OUTPUT_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/recon_generated"

    # Output directory for Nirvana generated images.
    NIRVANA_OUTPUT_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/nirvana_generated"

    # Directory for comparison outputs (excel/json summaries).
    COMPARE_SAVE_DIR = f"generated_results/{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/comparison_results"

    # Method names and generated-image directories used in compare.py.
    COMPARE_METHOD_DIRS = [
        ('Nirvana', NIRVANA_OUTPUT_DIR),
        ('MoDM', MODM_OUTPUT_DIR),
        ('ReDi', REDI_OUTPUT_DIR),
        ('Recon', RECON_OUTPUT_DIR),
        ('Ours', OURS_OUTPUT_DIR),
        ('OursMix', OURS_MIX_OUTPUT_DIR),
        ('OursGlobalNorm', OURS_GLOBAL_NORM_OUTPUT_DIR),
        ('OursMixGlobalNorm', OURS_MIX_GLOBAL_NORM_OUTPUT_DIR),
    ]

    # Compare metric switches.
    COMPARE_ENABLE_SSIM = False
    COMPARE_ENABLE_PSNR = False
    COMPARE_ENABLE_L2 = False
    COMPARE_ENABLE_CLIP_SIM = False
    COMPARE_ENABLE_CLIP_IMG_TEXT = True     # include "Org"
    COMPARE_ENABLE_SIGLIP2 = False          # include "Org"
    COMPARE_ENABLE_PICKSCORE = True         # include "Org"
    COMPARE_ENABLE_AESTHETICS = False
    COMPARE_ENABLE_FID = False
    COMPARE_ENABLE_KID = False
    COMPARE_ENABLE_IS = False               # include "Org"

    # SigLIP2 settings used in compare.py.
    COMPARE_SIGLIP2_CKPT = 'model_ckpt/siglip2-giant-opt-patch16-384'
    COMPARE_SIGLIP2_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    @classmethod
    def get_pipeline_class(cls):
        from pipeline import SD2Pipeline, SDXLPipeline, FLUX2Pipeline

        pipeline_map = {
            'SDXLPipeline': SDXLPipeline,
            'SD2.1Pipeline': SD2Pipeline,
            'FLUX2KleinPipeline': FLUX2Pipeline,
        }
        if cls.PIPELINE_CLASS_NAME not in pipeline_map:
            raise ValueError(
                f"Unsupported PIPELINE_CLASS_NAME={cls.PIPELINE_CLASS_NAME}. "
                f"Supported: {list(pipeline_map.keys())}"
            )
        return pipeline_map[cls.PIPELINE_CLASS_NAME]
