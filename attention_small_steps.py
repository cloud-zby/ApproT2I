import argparse
from config import Config

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=int, default=0, help='数据分片编号')
	parser.add_argument('--step', type=int, default=1000, help='每个分片的样本数')
	args = parser.parse_args()
	SPLIT = args.split
	STEP = args.step

	if Config.PIPELINE_CLASS_NAME == 'SDXLPipeline':
		from attention_small_steps_SDXL import run_attention_small_steps_sdxl
		run_attention_small_steps_sdxl(SPLIT, STEP)
	elif Config.PIPELINE_CLASS_NAME == 'SD2.1Pipeline':
		from attention_small_steps_SD2 import run_attention_small_steps_sd2
		run_attention_small_steps_sd2(SPLIT, STEP)
	elif Config.PIPELINE_CLASS_NAME == 'FLUX2KleinPipeline':
		from attention_small_steps_FLUX import run_attention_small_steps_flux
		run_attention_small_steps_flux(SPLIT, STEP)
	else:
		raise ValueError(f"Unsupported PIPELINE_CLASS_NAME: {Config.PIPELINE_CLASS_NAME}")

if __name__ == '__main__':
	main()
