
from config import Config

def main():
	if Config.PIPELINE_CLASS_NAME == 'SDXLPipeline':
		from attention_small_steps_SDXL import run_attention_small_steps_sdxl
		run_attention_small_steps_sdxl()
	elif Config.PIPELINE_CLASS_NAME == 'SD2Pipeline':
		from attention_small_steps_SD2 import run_attention_small_steps_sd2
		run_attention_small_steps_sd2()
	else:
		raise ValueError(f"Unsupported PIPELINE_CLASS_NAME: {Config.PIPELINE_CLASS_NAME}")

if __name__ == '__main__':
	main()
