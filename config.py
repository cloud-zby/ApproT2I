class Config:
    # Dataset file paths
    COCO_JSON_PATH = 'dataset/COCO2014_dataset.json'
    DIFFUSIONDB_PARQUET_PATH = 'dataset/DiffusionDB_dataset.parquet'
    PICKAPIC_PARQUET_PATH = 'dataset/PicaPic_dataset.parquet'

    # Dataset selector used by load_dataset().
    DATASET_NAME = 'coco'   # 'coco', 'pick-a-pic', or 'diffusiondb'

    # Pipeline class selector: 'SD2.1Pipeline' or 'SDXLPipeline' or 'FLUX2KleinPipeline'.
    PIPELINE_CLASS_NAME = 'FLUX2KleinPipeline'

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

    # Seed used in retrieval pipelines.
    RETRIEVE_SEED = 42

    # Retrieval metric for FAISS search: cosine, l2, or l1.
    DISTANCE_METRIC = 'cosine'.lower()

    # Top-k candidates used in prompt-based retrieval and recon fusion.
    TOP_K = 3

    # Noise scale added to the weighted latent in Recon method to enhance diversity in ReCon. Suggested range: 0.0 (no noise) to 0.1 (strong noise).
    RECON_ADD_NOISE_SCALE = 0.05

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
    [979, 959, 939, 919, 899, 
    879, 859, 839, 819, 799, 
    779, 759, 739, 719, 699, 
    679, 659, 639, 619, 599, 
    579, 559, 539, 519, 499, 
    479, 459, 439, 419, 399, 
    379, 359, 339, 319, 299, 
    279, 259, 239, 219, 199, 
    179, 159, 139, 119, 99, 
    79, 59, 39, 19, 0]
    SDXL
    
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
        'SD2.1Pipeline': [901, 801, 701, 601, 501],
        'SDXLPipeline': [901, 801, 701, 601, 501],
        'FLUX2KleinPipeline': [1000.0000,  988.4957,  974.4825,  957.0386,  934.7291, 905.1879, 864.2187,  803.6000,  704.7356,  514.7510],
    }
    TIMESTEPS_FOR_SMALL_STEPS = TIMESTEPS_FOR_SMALL_STEPS_DICT[PIPELINE_CLASS_NAME]

    # Number of self trajectory steps used before latent fusion.
    PRELIMINARY_STEPS = 4

    # Weight for self trajectory in Ours method. The final latent is a weighted sum of self and retrieved latents, where this weight controls the contribution of the self trajectory.
    OURS_SELF_WEIGHT = 0.5

    # Weight for self trajectory in Ours method. The final latent is a weighted sum of self and retrieved latents, where this weight controls the contribution of the self trajectory. Suggested range: 0.46 to 0.68.
    OURS_MIX_SELF_WEIGHT_MIN = 0.46
    OURS_MIX_SELF_WEIGHT_MAX = 0.68

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
