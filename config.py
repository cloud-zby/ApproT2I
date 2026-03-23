class Config:
    # Dataset selector used by load_dataset().
    DATASET_NAME = 'coco'   # 'coco', 'pick-a-pic', or 'diffusiondb'

    # Pipeline class selector: SDXLPipeline or SD2Pipeline.
    PIPELINE_CLASS_NAME = 'SDXLPipeline'

    # Model checkpoint paths for supported pipeline classes.
    MODEL_CKPT_PATHS = {
        'SDXLPipeline': './model_ckpt/stable-diffusion-xl-base-1.0',
        'SD2Pipeline': './model_ckpt/stable-diffusion-2-1',
    }

    # Unified model checkpoint path resolved from PIPELINE_CLASS_NAME.
    MODEL_CKPT_PATH = MODEL_CKPT_PATHS[PIPELINE_CLASS_NAME]

    # Seed for shuffling dataset indices before split processing.
    DATA_SHUFFLE_SEED = 19260817

    # Number of denoising steps for full trajectory generation.
    TRAJ_ALL_TOTAL_INFERENCE_STEPS = 50

    # Number of denoising steps when running short-step trajectories/attention capture.
    SMALL_STEPS_TOTAL_INFERENCE_STEPS = 10

    # Effective early steps to run in short-step mode.
    SMALL_STEPS_REAL_INFERENCE_STEPS = 5

    # Seed used in retrieval pipelines.
    RETRIEVE_SEED = 42

    # Retrieval metric for FAISS search: cosine, l2, or l1.
    DISTANCE_METRIC = 'cosine'.lower()

    # Top-k candidates used in prompt-based retrieval and recon fusion.
    TOP_K = 5

    # Reuse-info JSON path for image/prompt retrieval output and consumers.
    REUSE_INFO_PATH = f'reuse_infos_{DISTANCE_METRIC}.json'

    # Reuse-info JSON path for prompt-only top-k retrieval output and consumers.
    REUSE_INFO_BY_PROMPT_PATH = f'reuse_infos_{DISTANCE_METRIC}_by_prompt.json'

    # Number of denoising steps for all reuse-based generation methods.
    GENERATION_TOTAL_INFERENCE_STEPS = 50

    # Start timestep for reuse-based generation methods.
    GENERATION_BEGIN_TIMESTEP = 15

    # Number of self trajectory steps used before latent fusion.
    PRELIMINARY_STEPS = (GENERATION_BEGIN_TIMESTEP / GENERATION_TOTAL_INFERENCE_STEPS) * SMALL_STEPS_REAL_INFERENCE_STEPS
    assert PRELIMINARY_STEPS.is_integer(), f"PRELIMINARY_STEPS should be an integer, but got {PRELIMINARY_STEPS}. Please adjust SMALL_STEPS_REAL_INFERENCE_STEPS or GENERATION_BEGIN_TIMESTEP to ensure this."

    # Seed for torch.Generator in every pipeline sampling call.
    GENERATOR_SEED = 1024

    # Classifier-free guidance scale used in every generation call.
    GUIDANCE_SCALE = 7.5

    # Directory for full-trajectory images and latent trajectories.
    TRAJ_ALL_SAVE_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/org_images'

    # Directory for short-step images and latent trajectories.
    SMALL_STEPS_IMAGE_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/small_steps_images'

    # Directory for short-step attention matrices.
    SMALL_STEPS_ATTN_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/small_steps_attn'

    # Output directory for ReDi generated images.
    REDI_OUTPUT_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/redi_generated'

    # Output directory for MoDM generated images.
    MODM_OUTPUT_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/modm_generated'

    # Output directory for Ours generated images.
    OURS_OUTPUT_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/ours_generated'

    # Output directory for OursMix generated images.
    OURS_MIX_OUTPUT_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/ours_generated_mix'

    # Output directory for Recon generated images.
    RECON_OUTPUT_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/recon_generated'

    # Output directory for Nirvana generated images.
    NIRVANA_OUTPUT_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/nirvana_generated'

    # Directory for comparison outputs (excel/json summaries).
    COMPARE_SAVE_DIR = f'{PIPELINE_CLASS_NAME.replace('Pipeline', '')}/{DATASET_NAME}/comparison_results'

    # Method names and generated-image directories used in compare.py.
    COMPARE_METHOD_DIRS = [
        ('Nirvana', NIRVANA_OUTPUT_DIR),
        ('MoDM', MODM_OUTPUT_DIR),
        ('ReDi', REDI_OUTPUT_DIR),
        ('Recon', RECON_OUTPUT_DIR),
        ('Ours', OURS_OUTPUT_DIR),
        ('OursMix', OURS_MIX_OUTPUT_DIR),
        # ('OursNew', OURS_OUTPUT_DIR + "_new"),
        # ('OursNewMix', OURS_OUTPUT_DIR + "_new_mix"),
    ]

    @classmethod
    def get_pipeline_class(cls):
        from pipeline import SD2Pipeline, SDXLPipeline

        pipeline_map = {
            'SDXLPipeline': SDXLPipeline,
            'SD2Pipeline': SD2Pipeline,
        }
        if cls.PIPELINE_CLASS_NAME not in pipeline_map:
            raise ValueError(
                f"Unsupported PIPELINE_CLASS_NAME={cls.PIPELINE_CLASS_NAME}. "
                f"Supported: {list(pipeline_map.keys())}"
            )
        return pipeline_map[cls.PIPELINE_CLASS_NAME]
